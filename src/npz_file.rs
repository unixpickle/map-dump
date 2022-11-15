use std::{fmt::Write, fs::File, io::BufWriter, mem::take, path::Path};

use ndarray::{Array, Dimension};
use tokio::task::spawn_blocking;
use zip::{write::FileOptions, ZipWriter};

pub trait NumpyArrayElement {
    const DATA_SIZE: usize;
    const DATA_FORMAT: &'static str;

    fn encode<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()>;
}

macro_rules! array_element {
    ($dtype:ty, $data_size:literal, $data_format:literal) => {
        impl NumpyArrayElement for $dtype {
            const DATA_SIZE: usize = $data_size;
            const DATA_FORMAT: &'static str = $data_format;

            fn encode<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
                out.write_all(&self.to_le_bytes())
            }
        }
    };
}

array_element!(f32, 4, "<f4");
array_element!(f64, 8, "<f8");
array_element!(u8, 1, "<u1");
array_element!(u16, 2, "<u2");
array_element!(u32, 4, "<u4");
array_element!(u64, 8, "<u8");

pub trait NumpyArray: Sized {
    type Elem: NumpyArrayElement;
    type Iter: Iterator<Item = Self::Elem>;

    fn npy_shape(&self) -> Vec<usize>;
    fn npy_elements(self) -> Self::Iter;
}

impl<A: NumpyArrayElement> NumpyArray for A {
    type Elem = A;
    type Iter = <[A; 1] as IntoIterator>::IntoIter;

    fn npy_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    fn npy_elements(self) -> Self::Iter {
        [self].into_iter()
    }
}

impl<A: NumpyArrayElement, D: Dimension> NumpyArray for Array<A, D> {
    type Elem = A;
    type Iter = <Self as IntoIterator>::IntoIter;

    fn npy_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }

    fn npy_elements(self) -> Self::Iter {
        self.into_iter()
    }
}

impl<A: NumpyArrayElement> NumpyArray for Vec<A> {
    type Elem = A;
    type Iter = <Self as IntoIterator>::IntoIter;

    fn npy_shape(&self) -> Vec<usize> {
        vec![self.len()]
    }

    fn npy_elements(self) -> Self::Iter {
        self.into_iter()
    }
}

pub trait NumpyWritable: Sized {
    fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()>;
}

impl<A: NumpyArray> NumpyWritable for A {
    fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()> {
        out.write_all(&encode_header(
            <Self as NumpyArray>::Elem::DATA_FORMAT,
            &self.npy_shape(),
        ))?;
        for elem in self.npy_elements() {
            elem.encode(out)?;
        }
        Ok(())
    }
}

impl NumpyWritable for Vec<String> {
    fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()> {
        let unicode: Vec<Vec<u32>> = self
            .into_iter()
            .map(|x| x.chars().map(|x| x as u32).collect::<Vec<_>>())
            .collect();
        let max_result = unicode.iter().map(|x| x.len()).max();
        if let Some(max_len) = max_result {
            out.write_all(&encode_header(&format!("<U{}", max_len), &[unicode.len()]))?;
            for ustr in unicode {
                let mut byte_str = Vec::new();
                for ch in &ustr {
                    byte_str.extend(ch.to_le_bytes());
                }
                for _ in ustr.len()..max_len {
                    byte_str.extend([0, 0, 0, 0]);
                }
                out.write_all(&byte_str)?;
            }
            Ok(())
        } else {
            out.write_all(&encode_header("<U1", &[0]))
        }
    }
}

fn encode_header(format: &str, shape: &[usize]) -> Vec<u8> {
    let mut header_data = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': (",
        format
    );
    for (i, n) in shape.iter().enumerate() {
        if i > 0 {
            write!(header_data, " ").unwrap();
        }
        write!(header_data, "{},", n).unwrap();
    }
    write!(header_data, "), }}").unwrap();
    let mut header_bytes = header_data.into_bytes();
    while (11 + header_bytes.len()) % 64 != 0 {
        header_bytes.push(0x20);
    }
    header_bytes.push(0x0a);
    let mut all_bytes = Vec::new();
    all_bytes.push(0x93);
    all_bytes.extend("NUMPY".as_bytes());
    all_bytes.push(1);
    all_bytes.push(0);
    all_bytes.push((header_bytes.len() & 0xff) as u8);
    all_bytes.push(((header_bytes.len() >> 8) & 0xff) as u8);
    all_bytes.extend(header_bytes);
    all_bytes
}

pub struct NpzWriter {
    writer: Option<ZipWriter<File>>,
}

impl NpzWriter {
    pub async fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let path_clone = path.as_ref().to_owned();
        spawn_blocking(move || -> anyhow::Result<Self> { Self::new_blocking(&path_clone) }).await?
    }

    fn new_blocking(path: &Path) -> anyhow::Result<Self> {
        let file = File::create(path)?;
        let zw = ZipWriter::new(file);
        Ok(NpzWriter { writer: Some(zw) })
    }

    pub async fn write<T: NumpyWritable + Send + 'static>(
        &mut self,
        name: &str,
        t: T,
    ) -> std::io::Result<()> {
        if let Some(mut writer) = take(&mut self.writer) {
            let full_name = format!("{}.npy", name);
            self.writer = Some(
                spawn_blocking(move || -> std::io::Result<ZipWriter<File>> {
                    writer.start_file(&full_name, FileOptions::default())?;
                    let mut buffered = BufWriter::new(&mut writer);
                    t.write_npy(&mut buffered)?;
                    std::io::Write::flush(&mut buffered)?;
                    drop(buffered);
                    Ok(writer)
                })
                .await
                .or_else(|e| {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("failed to run write task: {}", e),
                    ))
                })??,
            );
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "a previous write has failed",
            ))
        }
    }

    pub async fn close(mut self) -> std::io::Result<()> {
        let writer = take(&mut self.writer);
        spawn_blocking(move || {
            if let Some(mut w) = writer {
                w.finish().ok();
            }
        })
        .await
        .or_else(|e| {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to run write task: {}", e),
            ))
        })
    }
}

impl Drop for NpzWriter {
    fn drop(&mut self) {
        let writer = take(&mut self.writer);
        spawn_blocking(move || {
            if let Some(mut w) = writer {
                w.finish().ok();
            }
        });
    }
}
