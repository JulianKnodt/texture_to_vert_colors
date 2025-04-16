use std::cmp::Ordering;
use std::iter::Peekable;

/// Merges two of the same iterator into a single iterator
/// Removing duplicates
#[derive(Debug, Clone)]
pub struct Merge<T, I: Iterator<Item = T>>(Peekable<I>, Peekable<I>);

impl<T, I: Iterator<Item = T>> Merge<T, I> {
    pub fn new(l: I, r: I) -> Self {
        Merge(l.peekable(), r.peekable())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side<T> {
    Left(T),
    Right(T),
    Both(T),
}

impl<T> Side<T> {
    pub fn into_inner(self) -> T {
        match self {
            Side::Left(v) => v,
            Side::Right(v) => v,
            Side::Both(v) => v,
        }
    }
    pub fn inner(&self) -> &T {
        match self {
            Side::Left(v) => v,
            Side::Right(v) => v,
            Side::Both(v) => v,
        }
    }
    pub fn is_both(&self) -> bool {
        matches!(self, Side::Both(_))
    }
}

impl<T, I> Iterator for Merge<T, I>
where
    I: Iterator<Item = T>,
    T: Ord,
{
    type Item = Side<T>;

    fn next(&mut self) -> Option<Self::Item> {
        use Side::*;
        let v = match (self.0.peek(), self.1.peek()) {
            (None, None) => return None,
            (Some(_), None) => Left(self.0.next()?),
            (None, Some(_)) => Right(self.1.next()?),
            (Some(a), Some(b)) => match a.cmp(b) {
                Ordering::Equal => {
                    let v = self.0.next();
                    assert!(v == self.1.next());
                    Both(v?)
                }
                Ordering::Less => Left(self.0.next()?),
                Ordering::Greater => Right(self.1.next()?),
            },
        };
        Some(v)
    }
}

#[test]
fn test_merge() {
    let a: &[usize] = &[0usize, 2, 4];
    let b: &[usize] = &[1usize, 3, 4, 5];
    assert_eq!(
        Merge::new(a.into_iter().copied(), b.into_iter().copied())
            .map(Side::into_inner)
            .collect::<Vec<_>>(),
        vec![0usize, 1, 2, 3, 4, 5]
    );
}
