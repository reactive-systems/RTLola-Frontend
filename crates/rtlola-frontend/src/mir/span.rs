#[cfg(feature = "spanned")]
mod spanned {
    use std::fmt::Display;

    use rtlola_reporting::Span;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
    pub struct Spanned<T> {
        pub inner: T,
        pub span: Span,
    }

    macro_rules! s {
		($t:ty) => {Spanned<$t>};
	}

    pub(crate) use s;

    impl<T> Spanned<T> {
        fn inner(&self) -> &T {
            &self.inner
        }

        fn into_inner(self) -> T {
            self.inner
        }

        fn span(&self) -> Span {
            self.span
        }
    }

    impl<T> AsRef<T> for Spanned<T> {
        fn as_ref(&self) -> &T {
            &self.inner
        }
    }

    impl<T: Display> Display for Spanned<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt(f)
        }
    }

    #[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
    pub struct MaybeSpanned<T> {
        pub inner: T,
        pub span: Option<Span>,
    }

    macro_rules! ms {
		($t:ty) => {MaybeSpanned<$t>};
	}

    pub(crate) use ms;

    impl<T> AsRef<T> for MaybeSpanned<T> {
        fn as_ref(&self) -> &T {
            &self.inner
        }
    }

    impl<T: Display> Display for MaybeSpanned<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt(f)
        }
    }

    macro_rules! spanned {
        ($t:expr, $s:expr) => {{
            use $crate::mir::Spanned;
            Spanned { inner: $t, span: $s }
        }};
    }

    pub(crate) use spanned;

    macro_rules! mspanned {
        ($t:expr, $s:expr) => {
            MaybeSpanned {
                inner: $t,
                span: Some($s),
            }
        };
        ($t:expr) => {
            MaybeSpanned {
                inner: $t,
                span: None,
            }
        };
    }

    pub(crate) use mspanned;

    macro_rules! inner {
        ($t:expr) => {
            $t.as_ref()
        };
    }

    pub(crate) use inner;
}

#[cfg(feature = "spanned")]
pub use spanned::*;

#[cfg(not(feature = "spanned"))]
#[macro_use]
mod unspanned {
    macro_rules! s {
        ($t:ty) => {
            $t
        };
    }

    pub(crate) use s;

    macro_rules! ms {
        ($t:ty) => {
            $t
        };
    }

    pub(crate) use ms;

    macro_rules! spanned {
        ($t:expr, $s:expr) => {
            $t
        };
    }

    pub(crate) use spanned;

    macro_rules! mspanned {
        ($t:expr, $s:expr) => {
            $t
        };
        ($t:expr) => {
            $t
        };
    }

    pub(crate) use mspanned;

    macro_rules! inner {
        ($t:expr) => {
            (&$t)
        };
    }

    pub(crate) use inner;
}

#[cfg(not(feature = "spanned"))]
pub use unspanned::*;
