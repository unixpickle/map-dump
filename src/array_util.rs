use std::collections::HashMap;

use ndarray::{Array2, Axis, LinalgScalar};

use num_traits::{real::Real, Zero};
use serde_json::{Map, Value};

pub trait Numeric: LinalgScalar + Real + Into<Value> {}
impl<T> Numeric for T where T: LinalgScalar + Real + Into<Value> {}

pub fn dense_matrix_to_json<T: Numeric>(matrix: &Array2<T>) -> Value {
    matrix
        .outer_iter()
        .map(|x| x.iter().map(|x| *x).collect::<Vec<_>>())
        .collect::<Vec<_>>()
        .into()
}

pub fn vec_to_matrix<T: Numeric>(v: &Vec<Vec<T>>) -> Array2<T> {
    assert!(v.len() > 0);
    let inner_size = v[0].len();

    let mut res = Array2::<T>::zeros((v.len(), inner_size));
    for (i, xs) in v.iter().enumerate() {
        for (j, x) in xs.iter().enumerate() {
            *res.get_mut((i, j)).unwrap() = *x;
        }
    }
    res
}

pub fn normalize_rows<T: Numeric>(a: &Array2<T>) -> Array2<T> {
    return a / row_norms(a);
}

fn row_norms<T: Numeric>(a: &Array2<T>) -> Array2<T> {
    a.map(|x| (*x) * (*x))
        .sum_axis(Axis(1))
        .map(|x| x.sqrt())
        .into_shape((a.shape()[0], 1))
        .unwrap()
}

#[derive(Clone, Debug)]
pub struct SparseMatrix<T: Numeric> {
    coord_to_data: HashMap<(usize, usize), T>,
    size: (usize, usize),
}

impl<T: Numeric> SparseMatrix<T> {
    pub fn zeros(size: (usize, usize)) -> Self {
        return SparseMatrix {
            coord_to_data: HashMap::new(),
            size,
        };
    }

    pub fn add_entry(&mut self, idx: (usize, usize), value: T) {
        if !value.is_zero() {
            let ptr = self.coord_to_data.entry(idx).or_insert(Zero::zero());
            *ptr = *ptr + value;
        }
    }

    pub fn into_json(self, sparse: bool) -> Value {
        if sparse {
            self.into()
        } else {
            dense_matrix_to_json(&(&self).into())
        }
    }
}

impl<T: Numeric> std::ops::AddAssign<&SparseMatrix<T>> for SparseMatrix<T> {
    fn add_assign(&mut self, rhs: &SparseMatrix<T>) {
        for ((row, col), val) in rhs.coord_to_data.iter() {
            self.add_entry((*row, *col), *val);
        }
    }
}

impl<T: Numeric> From<&Array2<T>> for SparseMatrix<T> {
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

impl<T: Numeric> From<SparseMatrix<T>> for Value {
    fn from(mat: SparseMatrix<T>) -> Value {
        let cap = mat.coord_to_data.len();
        let mut rows = Vec::with_capacity(cap);
        let mut cols = Vec::with_capacity(cap);
        let mut values = Vec::with_capacity(cap);
        for ((row, col), value) in mat.coord_to_data.into_iter() {
            if !value.is_zero() {
                rows.push(row);
                cols.push(col);
                values.push(value);
            }
        }
        Value::Object(Map::from_iter([
            ("indices".to_owned(), Value::from(vec![rows, cols])),
            ("values".to_owned(), Value::from(values)),
        ]))
    }
}

impl<T: Numeric> From<&SparseMatrix<T>> for Array2<T> {
    fn from(mat: &SparseMatrix<T>) -> Array2<T> {
        let mut res = Array2::<T>::zeros(mat.size);
        for ((row, col), val) in mat.coord_to_data.iter() {
            res[(*row, *col)] = *val;
        }
        res
    }
}
