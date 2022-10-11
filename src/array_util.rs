use ndarray::{Array2, Axis, LinalgScalar};

use serde_json::{Map, Value};

pub fn matrix_to_json<T: LinalgScalar + Into<Value>>(matrix: &Array2<T>, sparse: bool) -> Value {
    if !sparse {
        matrix
            .outer_iter()
            .map(|x| x.iter().map(|x| *x).collect::<Vec<_>>())
            .collect::<Vec<_>>()
            .into()
    } else {
        // Create a COO sparse matrix which is easily loadable
        // in PyTorch's sparse Tensor API.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();
        for ((row, col), value) in matrix.indexed_iter() {
            if !value.is_zero() {
                rows.push(row);
                cols.push(col);
                values.push(*value);
            }
        }
        Value::Object(Map::from_iter([
            ("indices".to_owned(), Value::from(vec![rows, cols])),
            ("values".to_owned(), Value::from(values)),
        ]))
    }
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
