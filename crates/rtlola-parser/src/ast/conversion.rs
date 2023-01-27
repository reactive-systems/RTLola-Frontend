#![allow(dead_code)]
use std::num::ParseIntError;
use std::str::FromStr;

use num::rational::Rational64 as Rational;
use num::traits::{CheckedMul, Inv};
use num::{FromPrimitive, One, Signed};
use uom::si::frequency::hertz;
use uom::si::rational64::{Frequency as UOM_Frequency, Time as UOM_Time};
use uom::si::time::second;

use super::{Expression, ExpressionKind, LitKind, Offset, TimeUnit};
use crate::ast::Literal;
use crate::parse::RtLolaParser;

pub(crate) type RationalType = i64;

impl Expression {
    /// Parses an expression into an offset used in [ExpressionKind::Offset]
    pub(crate) fn parse_offset(&self) -> Result<Offset, String> {
        if let Some(val) = self.parse_literal::<i16>() {
            Ok(Offset::Discrete(val))
        } else {
            // has to be a real-time expression
            let (val, unit) = match &self.kind {
                ExpressionKind::Lit(l) => {
                    match &l.kind {
                        LitKind::Numeric(val, Some(unit)) => (val, unit),
                        _ => return Err(format!("expected numeric value with unit, found `{l}`")),
                    }
                },
                _ => return Err(format!("expected numeric value with unit, found `{self}`")),
            };
            Ok(Offset::RealTime(
                RtLolaParser::parse_rational(val)?,
                TimeUnit::from_str(unit)?,
            ))
        }
    }

    /// Parses an expression into a duration for a discrete window [ExpressionKind::DiscreteWindowAggregation]
    pub fn parse_discrete_duration(&self) -> Result<u64, String> {
        match &self.kind {
            ExpressionKind::Lit(l) => {
                match &l.kind {
                    LitKind::Numeric(val, None) => val.parse().map_err(|err: ParseIntError| err.to_string()),
                    _ => Err(format!("expected numeric value without unit, found `{l}`")),
                }
            },
            _ => Err(format!("expected numeric value without unit, found `{self}`")),
        }
    }

    /// Parses an expression into a duration with a given unit of time
    pub(crate) fn parse_duration(&self) -> Result<UOM_Time, String> {
        let (val, unit) = match &self.kind {
            ExpressionKind::Lit(l) => {
                match &l.kind {
                    LitKind::Numeric(val, Some(unit)) => (RtLolaParser::parse_rational(val)?, unit),
                    _ => return Err(format!("expected numeric value with unit, found `{l}`")),
                }
            },
            _ => return Err(format!("expected numeric value with unit, found `{self}`")),
        };

        match unit.as_str() {
            "ns" | "μs" | "us" | "ms" | "s" | "min" | "h" | "d" | "w" | "a" => {
                use uom::si::time::*;
                let factor = match unit.as_str() {
                    "ns" => UOM_Time::new::<nanosecond>(Rational::one()),
                    "μs" | "us" => UOM_Time::new::<microsecond>(Rational::one()),
                    "ms" => UOM_Time::new::<millisecond>(Rational::one()),
                    "s" => UOM_Time::new::<second>(Rational::one()),
                    "min" => UOM_Time::new::<minute>(Rational::one()),
                    "h" => UOM_Time::new::<hour>(Rational::one()),
                    "d" => UOM_Time::new::<day>(Rational::one()),
                    "w" => UOM_Time::new::<day>(Rational::from_u64(7).unwrap()),
                    "a" => UOM_Time::new::<day>(Rational::from_u64(365).unwrap()),
                    u => unreachable!("'{}' should not have been catched by outer match", u),
                };
                let factor = factor.get::<second>();
                let duration = match val.checked_mul(&factor) {
                    Some(d) => d,
                    _ => {
                        return Err(format!(
                            "parsing duration failed: rational {val}*{factor} does not fit into Rational64"
                        ))
                    },
                };
                Ok(UOM_Time::new::<second>(duration))
            },
            u => Err(format!("expected duration unit, found `{u}`")),
        }
    }

    /// Parses an expression into a frequency.
    /// Expression must be a positive numeric value with Hz unit.
    pub fn parse_frequency(&self) -> Result<UOM_Frequency, String> {
        let (val, unit) = match &self.kind {
            ExpressionKind::Lit(l) => {
                match &l.kind {
                    LitKind::Numeric(val, Some(unit)) => (RtLolaParser::parse_rational(val)?, unit),
                    _ => return Err(format!("expected numeric value with unit, found `{l}`")),
                }
            },
            _ => return Err(format!("expected numeric value with unit, found `{self}`")),
        };

        if !val.is_positive() {
            return Err("frequencies have to be positive".to_string());
        }

        assert!(val.is_positive());

        match unit.as_str() {
            "μHz" | "uHz" | "mHz" | "Hz" | "kHz" | "MHz" | "GHz" => {
                use uom::si::frequency::*;
                let factor = match unit.as_str() {
                    "μHz" | "uHz" => UOM_Frequency::new::<microhertz>(Rational::one()),
                    "mHz" => UOM_Frequency::new::<millihertz>(Rational::one()),
                    "Hz" => UOM_Frequency::new::<hertz>(Rational::one()),
                    "kHz" => UOM_Frequency::new::<kilohertz>(Rational::one()),
                    "MHz" => UOM_Frequency::new::<megahertz>(Rational::one()),
                    "GHz" => UOM_Frequency::new::<gigahertz>(Rational::one()),
                    u => unreachable!("'{}' should not have been catched by outer match", u),
                };
                let factor = factor.get::<hertz>();
                let freq = match val.checked_mul(&factor) {
                    Some(f) => f,
                    _ => {
                        return Err(format!(
                            "parsing frequency failed: rational {val}*{factor} does not fit into Rational64",
                        ))
                    },
                };
                Ok(UOM_Frequency::new::<hertz>(freq))
            },
            u => Err(format!("expected frequency unit, found `{u}`")),
        }
    }

    /// Parses a frequency annotation.
    /// Expression must either be a frequency or a duration representing the period
    pub fn parse_freqspec(&self) -> Result<UOM_Frequency, String> {
        if let Ok(freq) = self.parse_frequency() {
            Ok(freq)
        } else if let Ok(period) = self.parse_duration() {
            let seconds = period.get::<second>();
            if seconds.is_positive() {
                Ok(UOM_Frequency::new::<hertz>(seconds.inv()))
            } else {
                Err(format!(
                    "duration of periodic stream specification must be positive, found `{period:#?}`"
                ))
            }
        } else {
            Err(format!("expected frequency or duration, found `{self}`"))
        }
    }

    /// Attempts to extract the numeric, constant, unit-less value out of an `Expression::Lit`.
    pub(crate) fn parse_literal<T>(&self) -> Option<T>
    where
        T: FromStr,
    {
        match &self.kind {
            ExpressionKind::Lit(l) => l.parse_numeric(),
            _ => None,
        }
    }
}

impl Literal {
    /// Parses a literal of kind numeric into its number representation i.e. u32
    pub(crate) fn parse_numeric<T>(&self) -> Option<T>
    where
        T: FromStr,
    {
        match &self.kind {
            LitKind::Numeric(val, unit) => {
                if unit.is_some() {
                    return None;
                }
                val.parse::<T>().ok()
            },
            _ => None,
        }
    }
}

impl Offset {
    /// Transforms a real-time offset into a [UOM_Time]
    pub fn to_uom_time(&self) -> Option<UOM_Time> {
        match self {
            Offset::Discrete(_) => None,
            Offset::RealTime(val, unit) => {
                let seconds = val * unit.to_uom_time().get::<second>();
                Some(UOM_Time::new::<second>(seconds))
            },
        }
    }
}

impl FromStr for TimeUnit {
    type Err = String;

    fn from_str(unit: &str) -> Result<Self, Self::Err> {
        match unit {
            "ns" => Ok(TimeUnit::Nanosecond),
            "μs" | "us" => Ok(TimeUnit::Microsecond),
            "ms" => Ok(TimeUnit::Millisecond),
            "s" => Ok(TimeUnit::Second),
            "min" => Ok(TimeUnit::Minute),
            "h" => Ok(TimeUnit::Hour),
            "d" => Ok(TimeUnit::Day),
            "w" => Ok(TimeUnit::Week),
            "a" => Ok(TimeUnit::Year),
            _ => Err(format!("unknown time unit `{unit}`")),
        }
    }
}

impl TimeUnit {
    /// Transforms a TimeUnit into a UOM_Time i.e. the number of seconds the timeunit spans.
    pub(crate) fn to_uom_time(self) -> UOM_Time {
        let f = match self {
            TimeUnit::Nanosecond => {
                Rational::new(
                    RationalType::from_u64(1).unwrap(),
                    RationalType::from_u64(10_u64.pow(9)).unwrap(),
                )
            },
            TimeUnit::Microsecond => {
                Rational::new(
                    RationalType::from_u64(1).unwrap(),
                    RationalType::from_u64(10_u64.pow(6)).unwrap(),
                )
            },
            TimeUnit::Millisecond => {
                Rational::new(
                    RationalType::from_u64(1).unwrap(),
                    RationalType::from_u64(10_u64.pow(3)).unwrap(),
                )
            },
            TimeUnit::Second => Rational::from_u64(1).unwrap(),
            TimeUnit::Minute => Rational::from_u64(60).unwrap(),
            TimeUnit::Hour => Rational::from_u64(60 * 60).unwrap(),
            TimeUnit::Day => Rational::from_u64(60 * 60 * 24).unwrap(),
            TimeUnit::Week => Rational::from_u64(60 * 60 * 24 * 7).unwrap(),
            TimeUnit::Year => Rational::from_u64(60 * 60 * 24 * 365).unwrap(),
        };
        UOM_Time::new::<second>(f)
    }
}

impl Expression {
    /// Tries to resolve a tuple index access
    fn get_expr_from_tuple(&self, idx: usize) -> Option<&Expression> {
        use ExpressionKind::*;
        match &self.kind {
            Tuple(entries) => Some(&entries[idx]),
            _ => None,
        }
    }

    /// A recursive iterator over an `Expression` tree
    /// Inspired by https://amos.me/blog/2019/recursive-iterators-rust/
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &Expression> + 'a> {
        use ExpressionKind::*;
        match &self.kind {
            Lit(_) | Ident(_) | MissingExpression => Box::new(std::iter::once(self)),
            Unary(_, inner)
            | Field(inner, _)
            | StreamAccess(inner, _)
            | Offset(inner, _)
            | ParenthesizedExpression(_, inner, _) => Box::new(std::iter::once(self).chain(inner.iter())),
            Binary(_, left, right)
            | Default(left, right)
            | DiscreteWindowAggregation {
                expr: left,
                duration: right,
                ..
            }
            | SlidingWindowAggregation {
                expr: left,
                duration: right,
                ..
            } => Box::new(std::iter::once(self).chain(left.iter()).chain(right.iter())),
            Ite(cond, normal, alternative) => {
                Box::new(
                    std::iter::once(self)
                        .chain(cond.iter())
                        .chain(normal.iter())
                        .chain(alternative.iter()),
                )
            },
            Tuple(entries) | Function(_, _, entries) => {
                Box::new(std::iter::once(self).chain(entries.iter().flat_map(|entry| entry.iter())))
            },
            Method(base, _, _, arguments) => {
                Box::new(
                    std::iter::once(self)
                        .chain(base.iter())
                        .chain(arguments.iter().flat_map(|entry| entry.iter())),
                )
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use num::ToPrimitive;

    use super::*;
    use crate::ast::{Literal, NodeId, Span};

    #[test]
    fn test_parse_rational() {
        macro_rules! check_on {
            ($f:expr) => {
                let f_string = format!("{}", $f);
                let f = f_string.parse::<f64>().unwrap();
                let was = super::RtLolaParser::parse_rational(f_string.as_str())
                    .unwrap_or_else(|e| panic!("parsing failed: {}", e));
                assert_eq!(was, Rational::from_f64(f).unwrap());
            };
        }
        check_on!(0);
        check_on!(42);
        check_on!(-1);
        check_on!(0.1);
        check_on!(42.12);
        check_on!(-1.123);
        check_on!(0.1e-0);
        check_on!(42.12e+1);
        check_on!(-1.123e-2);
    }

    fn time_spec_int(val: &str, unit: &str) -> Duration {
        let expr = Expression::new(
            NodeId::new(32),
            ExpressionKind::Lit(Literal::new_numeric(
                NodeId::new(24),
                val,
                Some(unit.to_string()),
                Span::Unknown,
            )),
            Span::Unknown,
        );
        let freq = expr.parse_freqspec().unwrap();
        let period = UOM_Time::new::<second>(freq.get::<hertz>().inv());
        Duration::from_nanos(period.get::<uom::si::time::nanosecond>().to_integer().to_u64().unwrap())
    }

    #[test]
    fn test_time_spec_to_duration_conversion() {
        assert_eq!(time_spec_int("1", "s"), Duration::new(1, 0));
        assert_eq!(time_spec_int("2", "min"), Duration::new(2 * 60, 0));
        assert_eq!(time_spec_int("33", "h"), Duration::new(33 * 60 * 60, 0));
        assert_eq!(time_spec_int("12354", "ns"), Duration::from_nanos(12354));
        assert_eq!(time_spec_int("90351", "us"), Duration::from_nanos(90351 * 1_000));
        assert_eq!(time_spec_int("248", "ms"), Duration::from_nanos(248 * 1_000_000));
        assert_eq!(
            time_spec_int("29489232", "ms"),
            Duration::from_nanos(29_489_232 * 1_000_000)
        );
    }

    #[test]
    fn test_frequency_to_duration_conversion() {
        assert_eq!(time_spec_int("1", "Hz"), Duration::new(1, 0));
        assert_eq!(time_spec_int("10", "Hz"), Duration::new(0, 100_000_000));
        assert_eq!(time_spec_int("400", "uHz"), Duration::new(2_500, 0));
        assert_eq!(time_spec_int("20", "mHz"), Duration::new(50, 0));
    }
}
