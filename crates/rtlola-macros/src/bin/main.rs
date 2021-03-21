#[macro_use]
extern crate rtlola_macros;

fn main() {
    let hir_a = Hir { mode: ModeA { a: A {} } };
    hir_a.a(3);
    let hir_b = Hir { mode: ModeB { a: A {}, b: B {} } };
    hir_b.a(5);
    hir_b.b();

    let hir_c = Hir { mode: ModeC { a: A {}, b: B {}, c: C {} } };
    hir_c.a(5);
    hir_c.b();
    hir_c.c();
    foo(hir_c);
}

fn foo<M: HirMode + TraitB>(ir: Hir<M>) {
    ir.b()
}

struct Hir<M: HirMode> {
    mode: M,
}

trait HirMode {}

#[covers_functionality(TraitA, a)]
#[derive(HirMode)]
struct ModeA {
    a: A,
}

struct A {}

#[mode_functionality]
trait TraitA {
    fn a(&self, x: u32);
}

impl TraitA for A {
    fn a(&self, _x: u32) {
        println!("I'm an A.");
    }
}

#[covers_functionality(TraitA, a)]
#[covers_functionality(TraitB, b)]
#[derive(HirMode)]
struct ModeB {
    a: A,
    b: B,
}

struct B {}

#[mode_functionality]
trait TraitB {
    fn b(&self);
}

impl TraitB for B {
    fn b(&self) {
        println!("I'm a B.");
    }
}

#[covers_functionality(TraitA, a)]
#[covers_functionality(TraitB, b)]
#[covers_functionality(TraitC, c)]
#[derive(HirMode)]
struct ModeC {
    a: A,
    b: B,
    c: C,
}

struct C {}

#[mode_functionality]
trait TraitC {
    fn c(&self);
}

impl TraitC for C {
    fn c(&self) {
        println!("I'm a C.");
    }
}
