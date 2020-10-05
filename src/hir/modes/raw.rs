use std::{collections::HashMap, rc::Rc};

use crate::common_ir::StreamReference as SRef;
use crate::{
    ast::{self, Ast},
    hir::Hir,
    hir::Input,
    hir::Output,
    hir::Trigger,
};

use super::{IrExpression, Raw};

impl From<Ast> for Hir<Raw> {
    fn from(ast: Ast) -> Hir<Raw> {
        let Ast {
            imports: _, // todo
            constants,
            inputs,
            outputs,
            trigger,
            type_declarations: _,
        } = ast;

        let mut expressions: HashMap<SRef, ast::Expression> = HashMap::new();
        let mut hir_outputs = vec![];
        for (ix, o) in outputs.into_iter().enumerate() {
            let sr = SRef::OutRef(ix);
            let ast::Output { expression, name, .. } =
                Rc::try_unwrap(o).expect("other strong references should be dropped now");
            hir_outputs.push(Output { name: name.name, sr });
            expressions.insert(sr, expression);
        }
        let hir_outputs = hir_outputs;
        let mut hir_triggers = vec![];
        for (ix, t) in trigger.into_iter().enumerate() {
            let sr = SRef::OutRef(hir_outputs.len() + ix);
            let ast::Trigger { message, name, expression, .. } =
                Rc::try_unwrap(t).expect("other strong references should be dropped now");
            hir_triggers.push(Trigger::new(name, message, sr));
            expressions.insert(sr, expression);
        }
        let hir_triggers = hir_triggers;
        let hir_inputs: Vec<Input> = inputs
            .into_iter()
            .enumerate()
            .map(|(ix, i)| Input {
                name: Rc::try_unwrap(i).expect("other strong references should be dropped now").name.name,
                sr: SRef::InRef(ix),
            })
            .collect();

        let mode = Raw {
            constants: constants
                .into_iter()
                .map(|c| Rc::try_unwrap(c).expect("other strong references should be dropped now"))
                .collect(),
            expressions,
        };

        let next_input_ref = hir_inputs.len();
        let next_output_ref = hir_outputs.len() + hir_triggers.len();
        Hir { inputs: hir_inputs, outputs: hir_outputs, triggers: hir_triggers, next_input_ref, next_output_ref, mode }
    }
}

impl Hir<Raw> {
    pub(crate) fn replace_expressions(self) -> Hir<IrExpression> {
        unimplemented!()
    }
}
