use std::sync::Arc;

use tokio::sync::Mutex;

pub struct TaskQueue<T: Send>(Arc<Mutex<Vec<T>>>);

impl<T: Send> Clone for TaskQueue<T> {
    fn clone(&self) -> TaskQueue<T> {
        TaskQueue(self.0.clone())
    }
}

impl<T: Send, I: IntoIterator<Item = T>> From<I> for TaskQueue<T> {
    fn from(x: I) -> TaskQueue<T> {
        TaskQueue(Arc::new(Mutex::new(x.into_iter().collect())))
    }
}

impl<T: Send> TaskQueue<T> {
    pub async fn pop(&self) -> Option<T> {
        self.0.lock().await.pop()
    }

    pub async fn len(&self) -> usize {
        self.0.lock().await.len()
    }
}
