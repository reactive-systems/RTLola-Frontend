//! End-to-end tests of the RTLola frontend
use super::*;

fn parse(spec: &str) -> Result<RtLolaMir, String> {
    let cfg = ParserConfig::for_string(String::from(spec));
    crate::parse(cfg).map_err(|e| format!("parsing failed with the following error:\n{:?}", e))
}

#[test]
fn fuzzed_unknown_unit() {
    assert!(parse("input a: Int\ninput b: Int\n\noutput c @0a := a + b.hold().defaults(to: 0)\n\ntrigger c > 2 \"c is too large\"").is_err());
    assert!(parse("input a: Int\ninput b: Int\n\noutput c @8a := a + b.hold().defaults(to: 0)\n\ntrigger c > 2 \"c is too large\"").is_err());
}

#[test]
fn fuzzed_bad_floating() {
    assert!(parse("output d @ 2.5ez := 0").is_err());
}

#[test]
fn fuzzed_denominator_eq_0() {
    assert!(parse(
        "input a: Int\n\noutput b @ 00Hz := a.hold().defaults(to:10)\noutput c @ 5Hz := a.hold().defaults(to:10)"
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput b @ -0Hz := a.hold().defaults(to:10)\noutput c @ 5Hz := a.hold().defaults(to:10)"
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput b @ +0Hz := a.hold().defaults(to:10)\noutput c @ 5Hz := a.hold().defaults(to:10)"
    )
    .is_err());
}

#[test]
fn fuzzed_lowering_bad_assumptions1() -> Result<(), String> {
    parse(
        "input a: Int\n\noutput b @ 10Hz := c.hold().defaults(to:10)\noutput c @ 5Hz := a.hold().defaults(to:10)\n\n\n",
    )?;
    assert!(parse("input a: Int\n\noutput s @ 10Hz := a.aggregate(over: 0.1s, using: sum)\noutput c @ 10Hz := a.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1s, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral\n.defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\"").is_err());
    Ok(())
}

#[test]
fn fuzzed_lowering_bad_assumptions2() {
    assert!(parse("input a: Int\noutput d := a.get().defaults(to:1111111111111111111111111110)\n\ntrigger d == 2 || d == 3 \"valuY used\"\ntrigger d ==10 \"default used\"").is_err());
}

#[test]
fn fuzzed_lowering_bad_assumptions3() {
    assert!(parse("input a: Int\n\noutput b := a.offset(by:-000).defaults(to:300)\n\ntrigger b == 100 \"b is 100\"\ntrigger b == 99 \"b is 99\"\ntrigger b == 300 \"default used\"").is_err());
}

#[test]
fn fuzzed_negative_frequency() {
    assert!(parse(
        "input a: Int\n\noutput b @-10Hz := a.hold().defaults(to:10)\noutput c @ 5Hz := a.hold().defaults(to:10)\n\n\n"
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput b @ 10Hz := a.hold().defaults(to:10)\noutput c @-5Hz := a.hold().defaults(to:10)\n\n\n"
    )
    .is_err());
}

#[ignore = "Future offsets not implemented"]
#[test]
fn fuzzed_memory_analysis_unimplemented() {
    assert!(parse(
        "input a: Int\n\noutput b := a.offset(by:
100).defaults(to:300)\n\ntrigger b == 100 \"b is 100\"\ntrigger b == 99 \"b is 99\"\ntrigger b == 300 \"default used\""
    )
    .is_ok());
    assert!(parse(
        "input a: Int\n\noutput b := a.offset(by:5100).defaults(to:300)\n\ntrigger b == 100 \"b is 100\"\ntrigger b == 99 \"b is 99\"\ntrigger b == 300 \"default used\""
    )
    .is_ok());
    assert!(parse(
        "input a: Int\n\noutput b := a.offset(by:+100).defaults(to:300)\n\ntrigger b == 100 \"b is 100\"\ntrigger b == 99 \"b is 99\"\ntrigger b == 300 \"default used\""
    )
    .is_ok());
}

#[test]
fn fuzzed_dependency_analysis_unimplemented() {
    assert!(parse(
        "input a: Int\n\noutput s @ 10Hz := a.aggregate(over: 0.1s, using: sum)\noutput c @ 10Hz := c.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1s, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral).defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\""
    )
    .is_err());
}

#[test]
fn fuzzed_aggregation() {
    assert!(parse(
        "input a: Int\n\noutput s @ 10Hz := a.aggregate(over:!0.1s, using: sum)\noutput c @ 10Hz := a.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1s, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral).defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\""
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput s @ 10Hz := a.aggregate(over: 0>1s, using: sum)\noutput c @ 10Hz := a.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1s, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral).defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\""
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput s @ 10Hz := a.aggregate(over: 0*1s, using: sum)\noutput c @ 10Hz := a.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1s, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral).defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\""
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput s @ 10Hz := a.aggregate(over: 0.13, using: sum)\noutput c @ 10Hz := a.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1s, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral).defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\""
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput s @ 10Hz := a.aggregate(over: 0.1S, using: sum)\noutput c @ 10Hz := a.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1s, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral).defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\""
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput s @ 10Hz := a.aggregate(over: 0.=s, using: sum)\noutput c @ 10Hz := a.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1s, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral).defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\""
    )
    .is_err());
    assert!(parse(
        "input a: Int\n\noutput s @ 10Hz := a.aggregate(over: 0.1s, using: sum)\noutput c @ 10Hz := a.aggregate(over: 0.1s, using: count)\noutput av @ 10Hz := a.aggregate(over: 0.1O, using: avg).defaults(to: 10)\n//output i @ 10Hz := a.aggregate(over: 0.1s, using: integral).defaults(to: 10)\n\ntrigger c == 1 \"count is 1\"\ntrigger c == 2 \"count is 2\""
    )
    .is_err());
}

#[test]
fn fuzzed_type_checker_bad_assumptions() {
    assert!(parse("input a: Int\ninput b: Int\n\noutput c :=0a + b\n\ntrigger c . 2 \"c is too large\"").is_err());
    assert!(parse("input a: Int\ninput b: Int\n\noutput c :=!a + b\n\ntrigger c . 2 \"c is too large\"").is_err());
    assert!(parse("input a: Int\ninput b: Int\n\noutput c := ! + b\n\ntrigger c . 2 \"c is too large\"").is_err());
    assert!(parse("input a: Int\ninput b: Int\n\noutput c := a +0b\n\ntrigger c . 2 \"c is too large\"").is_err());
    assert!(parse("input a: Int\ninput b: Int\n\noutput c := a&+ b\n\ntrigger c . 2 \"c is too large\"").is_err());
    assert!(parse("input a: Int\ninput b: Int\n\noutput c @0a := a +!b.hold().defaults(to: 0)\n\ntrigger c > 2 \"c is too large\"").is_err());
    assert!(parse("input a: Int\ninput b: Int\n\noutput c @0a := a + b.triggerdefaults(to: 0)\n\ntrigger c > 2 \"c is too large\"").is_err());
}

#[test]
fn fuzzed_type_checker_no_whitespace() {
    assert!(parse("constantc : Bool := false").is_err());
    assert!(parse("constant c : Bool := false").is_ok());
    assert!(parse("inputa: Int").is_err());
    assert!(parse("input a: Int").is_ok());
    assert!(parse("input a: Int\noutputb := a").is_err());
    assert!(parse("input a: Int\noutput b := a").is_ok());
    assert!(parse("input a: Int\ntriggera > 0").is_err());
    assert!(parse("input a: Int\ntrigger a > 0").is_ok());
}

#[test]
fn fuzzed_type_checker_tuple() {
    assert!(parse("output out: (Int, Bool) @1Hz := (1, false)").is_ok());
    assert!(parse("output out: (Int, Bool) @1Hz := ((1), false)").is_ok());
}

#[test]
fn fuzzed_big_literal() {
    assert!(parse("output o := 111111111111111111111111111").is_err());
}

#[test]
fn fuzzed_big_frequency() {
    assert!(parse("output X @ 15111111111111MHz := 1").is_err());
    assert!(parse("output X @ 15111111111111mHz := 1").is_ok());
}

#[test]
fn fuzzed_invalid_activation_condition() {
    assert!(parse("input a: Int output x @x := a.hold().defaults(to: 0)").is_err());
}

#[ignore = "real-time offsets not implemented, yet"]
#[test]
fn fuzzed_big_realtime_offset() {
    assert!(parse("output a: Int8 @0.5Hz := 1 output b: Int8 @1Hz := a[-1w].defaults(to: 0)").is_err());
}

#[test]
fn fuzzed_activation_condition_greedy_lookup() {
    assert!(parse("output a: Int8 @b := 0 output b: Int8 @ 1Hz := 0").is_err());
    assert!(parse("output b: Int8 @ 1Hz := 0 output a: Int8 @b := 0").is_err());
}

#[test]
fn fuzzed_tuple_access_on_steriods() {
    assert!(parse("output count := count.8>(-1).defaulM(0) . 1").is_err());
}

#[test]
fn min_max() -> Result<(), String> {
    parse("import math\n input a: Int\n input b: Int\n output min_max := min<Int>(max<Int>(a,b), b)")?;
    Ok(())
}

#[test]
fn min_incompatible() {
    assert!(parse("import math\n input a: Int\n input b: Float64\n output minres := min(a, b)").is_err());
}

#[test]
fn test_float16() {
    assert!(parse("input in: Float16\noutput count := in + 3.5").is_ok());
}

#[ignore = "Future offsets not implemented, yet"]
#[test]
fn future_offset() {
    assert!(parse("input a: Int8\noutput b := a.offset(by: 1).defaults(to: 3)").is_ok());
}
