use std::collections::HashMap;

use ndarray::{Array2, Axis, LinalgScalar};

use num_traits::Zero;
use serde_json::{Map, Value};

pub fn dense_matrix_to_json<T: LinalgScalar + Into<Value>>(matrix: &Array2<T>) -> Value {
    matrix
        .outer_iter()
        .map(|x| x.iter().map(|x| *x).collect::<Vec<_>>())
        .collect::<Vec<_>>()
        .into()
}

pub fn vec_to_matrix(v: &Vec<Vec<f32>>) -> Array2<f32> {
    assert!(v.len() > 0);
    let inner_size = v[0].len();

    let mut res = Array2::<f32>::zeros((v.len(), inner_size));
    for (i, xs) in v.iter().enumerate() {
        for (j, x) in xs.iter().enumerate() {
            *res.get_mut((i, j)).unwrap() = *x;
        }
    }
    res
}

pub fn normalize_rows(a: &Array2<f32>) -> Array2<f32> {
    return a / row_norms(a);
}

fn row_norms(a: &Array2<f32>) -> Array2<f32> {
    a.map(|x| x * x)
        .sum_axis(Axis(1))
        .map(|x| x.sqrt())
        .into_shape((a.shape()[0], 1))
        .unwrap()
}

#[derive(Clone, Debug)]
pub struct SparseMatrix<T: LinalgScalar + Into<Value>> {
    coord_to_data: HashMap<(usize, usize), T>,
    size: (usize, usize),
}

impl<T: LinalgScalar + Into<Value>> SparseMatrix<T> {
    pub fn zeros(size: (usize, usize)) -> Self {
        return SparseMatrix {
            coord_to_data: HashMap::new(),
            size,
        };
    }

    pub fn dense(&self) -> Array2<T> {
        let mut res = Array2::<T>::zeros(self.size);
        for ((row, col), val) in self.coord_to_data.iter() {
            res[(*row, *col)] = *val;
        }
        res
    }

    pub fn add_entry(&mut self, idx: (usize, usize), value: T) {
        if !value.is_zero() {
            let ptr = self.coord_to_data.entry(idx).or_insert(Zero::zero());
            *ptr = *ptr + value;
        }
    }

    pub fn to_json(&self, sparse: bool) -> Value {
        if sparse {
            self.into()
        } else {
            dense_matrix_to_json(&self.dense())
        }
    }
}

impl<T: LinalgScalar + Into<Value>> std::ops::AddAssign<&SparseMatrix<T>> for SparseMatrix<T> {
    fn add_assign(&mut self, rhs: &SparseMatrix<T>) {
        for ((row, col), val) in rhs.coord_to_data.iter() {
            self.add_entry((*row, *col), *val);
        }
    }
}

impl<T: LinalgScalar + Into<Value>> From<&Array2<T>> for SparseMatrix<T> {
    fn from(matrix: &Array2<T>) -> SparseMatrix<T> {
        let mut res = SparseMatrix::zeros(matrix.dim());
        for ((row, col), value) in matrix.indexed_iter() {
            if !value.is_zero() {
                res.coord_to_data.insert((row, col), *value);
            }
        }
        res
    }
}

impl<T: LinalgScalar + Into<Value>> From<&SparseMatrix<T>> for Value {
    fn from(mat: &SparseMatrix<T>) -> Value {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();
        for ((row, col), value) in mat.coord_to_data.iter() {
            if !value.is_zero() {
                rows.push(*row);
                cols.push(*col);
                values.push(*value);
            }
        }
        Value::Object(Map::from_iter([
            ("indices".to_owned(), Value::from(vec![rows, cols])),
            ("values".to_owned(), Value::from(values)),
        ]))
    }
}
