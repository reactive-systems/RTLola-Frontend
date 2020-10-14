/*!
This module describes intermediate representations that are use in the high level intermediate representation and in the mid level intermediate representation.
*/
#![allow(dead_code)]
use std::time::Duration;
use uom::si::rational64::Frequency as UOM_Frequency;
use uom::si::rational64::Time as UOM_Time;

pub mod print;

/// Wrapper for output streams that are actually triggers.  Provides additional information specific to triggers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Trigger {
    /// The trigger message that is supposed to be conveyed to the user if the trigger reports a violation.
    pub message: String,
    /// A reference to the output stream representing the trigger.
    pub reference: StreamReference,
}

/// This enum indicates how much memory is required to store a stream.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MemorizationBound {
    /// The required memory might exceed any bound.
    Unbounded,
    /// No less then the contained amount of stream entries does ever need to be stored.
    Bounded(u16),
}

impl PartialOrd for MemorizationBound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        use MemorizationBound::*;
        match (self, other) {
            (Unbounded, Unbounded) => None,
            (Bounded(_), Unbounded) => Some(Ordering::Less),
            (Unbounded, Bounded(_)) => Some(Ordering::Greater),
            (Bounded(b1), Bounded(b2)) => Some(b1.cmp(&b2)),
        }
    }
}

/// This data type provides information regarding how much data a stream needs to have access to from another stream.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Tracking {
    /// Need to store every single value of a stream
    All(StreamReference),
    /// Need to store `num` values of `trackee`, evicting/add a value every `rate` time units.
    Bounded {
        /// The stream that will be tracked.
        trackee: StreamReference,
        /// The number of values that will be accessed.
        num: u128,
        /// The duration in which values might be accessed.
        rate: Duration,
    },
}

/// Wrapper for output streams providing additional information specific to timedriven streams.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct TimeDrivenStream {
    /// A reference to the stream that is specified.
    pub reference: StreamReference,
    /// The evaluation frequency of the stream.
    pub frequency: UOM_Frequency,
    /// The duration between two evaluation cycles.
    pub extend_rate: Duration,
    /// The period of the stream.
    pub period: UOM_Time,
}

/// Wrapper for output streams providing additional information specific to event-based streams.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct EventDrivenStream {
    /// A reference to the stream that is specified.
    pub reference: StreamReference,
}

/// Contains information regarding the dependency between two streams which occurs due to a lookup expression.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Dependency {
    /// The target of the lookup.
    pub stream: StreamReference,
    /// The offset of the lookup.
    pub offsets: Vec<Offset>,
}

/// Offset used in the lookup expression
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Offset {
    /// A strictly positive discrete offset, e.g., `4`, or `42`
    FutureDiscreteOffset(u32),
    /// A non-negative discrete offset, e.g., `0`, `-4`, or `-42`
    PastDiscreteOffset(u32),
    /// A positive real-time offset, e.g., `-3ms`, `-4min`, `-2.3h`
    FutureRealTimeOffset(Duration),
    /// A non-negative real-time offset, e.g., `0`, `4min`, `2.3h`
    PastRealTimeOffset(Duration),
}

/// TODO
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub struct Layer(usize);
impl Into<usize> for Layer {
    fn into(self) -> usize {
        self.0
    }
}

impl Layer {
    pub fn new(layer: usize) -> Self {
        Layer(layer)
    }
    pub fn inner(self) -> usize {
        self.0
    }
}

/////// Referencing Structures ///////

/// Allows for referencing a window instance.
#[derive(Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowReference(pub usize);

pub(crate) type WRef = WindowReference;

impl WindowReference {
    /// Provides access to the index inside the reference.
    pub fn idx(self) -> usize {
        self.0
    }
}

/// Allows for referencing an input stream within the specification.
pub type InputReference = usize;
/// Allows for referencing an output stream within the specification.
pub type OutputReference = usize;

/// Allows for referencing a stream within the specification.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StreamReference {
    /// References an input stream.
    InRef(InputReference),
    /// References an output stream.
    OutRef(OutputReference),
}

pub(crate) type SRef = StreamReference;

impl StreamReference {
    /// Returns the index inside the reference if it is an output reference.  Panics otherwise.
    pub fn out_ix(&self) -> usize {
        match self {
            StreamReference::InRef(_) => unreachable!(),
            StreamReference::OutRef(ix) => *ix,
        }
    }

    /// Returns the index inside the reference if it is an input reference.  Panics otherwise.
    pub fn in_ix(&self) -> usize {
        match self {
            StreamReference::OutRef(_) => unreachable!(),
            StreamReference::InRef(ix) => *ix,
        }
    }

    /// Returns the index inside the reference disregarding whether it is an input or output reference.
    pub fn ix_unchecked(&self) -> usize {
        match self {
            StreamReference::InRef(ix) | StreamReference::OutRef(ix) => *ix,
        }
    }

    pub fn is_input(&self) -> bool {
        match self {
            StreamReference::OutRef(_) => false,
            StreamReference::InRef(_) => true,
        }
    }

    pub fn is_output(&self) -> bool {
        match self {
            StreamReference::OutRef(_) => true,
            StreamReference::InRef(_) => false,
        }
    }
}

/// A trait for any kind of stream.
pub trait Stream {
    // TODO: probably not needed anymore
    /// Returns the evaluation laying in which the stream resides.
    fn eval_layer(&self) -> Layer;
    /// Indicates whether or not the stream is an input stream.
    fn is_input(&self) -> bool;
    // TODO: probably not needed anymore
    /// Indicates how many values need to be memorized.
    fn values_to_memorize(&self) -> MemorizationBound;
    /// Produces a stream references referring to the stream.
    fn as_stream_ref(&self) -> StreamReference;
}

////////// Implementations //////////

impl MemorizationBound {
    /// Produces the memory bound.  Panics if it is unbounded.
    pub fn unwrap(self) -> u16 {
        match self {
            MemorizationBound::Bounded(b) => b,
            MemorizationBound::Unbounded => {
                unreachable!("Called `MemorizationBound::unwrap()` on an `Unbounded` value.")
            }
        }
    }

    /// Produces the memory bound.  If it is unbounded, the default value will be returned.
    pub fn unwrap_or(self, dft: u16) -> u16 {
        match self {
            MemorizationBound::Bounded(b) => b,
            MemorizationBound::Unbounded => dft,
        }
    }
    /// Produces `Some(v)` if the memory bound is finite and `v` and `None` if it is unbounded.
    pub fn as_opt(self) -> Option<u16> {
        match self {
            MemorizationBound::Bounded(b) => Some(b),
            MemorizationBound::Unbounded => None,
        }
    }
}

impl PartialOrd for Offset {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        use Offset::*;
        match (self, other) {
            (PastDiscreteOffset(_), FutureDiscreteOffset(_))
            | (PastRealTimeOffset(_), FutureRealTimeOffset(_))
            | (PastDiscreteOffset(_), FutureRealTimeOffset(_))
            | (PastRealTimeOffset(_), FutureDiscreteOffset(_)) => Some(Ordering::Less),

            (FutureDiscreteOffset(_), PastDiscreteOffset(_))
            | (FutureDiscreteOffset(_), PastRealTimeOffset(_))
            | (FutureRealTimeOffset(_), PastDiscreteOffset(_))
            | (FutureRealTimeOffset(_), PastRealTimeOffset(_)) => Some(Ordering::Greater),

            (FutureDiscreteOffset(a), FutureDiscreteOffset(b)) => Some(a.cmp(b)),
            (PastDiscreteOffset(a), PastDiscreteOffset(b)) => Some(b.cmp(a)),

            (_, _) => unimplemented!(),
        }
    }
}

impl Ord for Offset {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// The size of a specific value in bytes.
#[derive(Debug, Clone, Copy)]
pub struct ValSize(pub u32); // Needs to be reasonable large for compound types.

impl From<u8> for ValSize {
    fn from(val: u8) -> ValSize {
        ValSize(u32::from(val))
    }
}

impl std::ops::Add for ValSize {
    type Output = ValSize;
    fn add(self, rhs: ValSize) -> ValSize {
        ValSize(self.0 + rhs.0)
    }
}
