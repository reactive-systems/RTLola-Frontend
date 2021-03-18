#[allow(unused_imports)]
use crate::common_ir::StreamReference as SRef;
#[allow(unused_imports)]
use crate::{
    ast::{self, Ast},
    hir::modes::ir_expr::annotated_type,
    hir::AnnotatedType,
    hir::Hir,
    hir::Input,
    hir::Output,
    hir::Parameter,
    hir::Trigger,
};
#[allow(unused_imports)]
use std::{collections::HashMap, rc::Rc};

use super::{IrExprMode, Raw};

impl From<Ast> for Hir<Raw> {
    fn from(_ast: Ast) -> Hir<Raw> {
        // let Ast {
        //     _imports: _, // todo
        //     _constants,
        //     _inputs,
        //     _outputs,
        //     _trigger,
        //     _type_declarations: _,
        // } = ast;
        /*
        let mut expressions: HashMap<SRef, ast::Expression> = HashMap::new();
        let mut template_specs: HashMap<SRef, ast::TemplateSpec> = HashMap::new();
        let mut hir_outputs = vec![];
        for (ix, o) in outputs.into_iter().enumerate() {
            let sr = SRef::OutRef(ix);
            let ast::Output { expression, name, params, template_spec, ty, .. } =
                Rc::try_unwrap(o).expect("other strong references should be dropped now");
            let params: Vec<Parameter> = params
                .iter()
                .enumerate()
                .map(|(ix, p)| Parameter { name: p.name.name.clone(), annotated_type: annotated_type(&p.ty), idx: ix })
                .collect();
            let annotated_type = annotated_type(&ty);
            //hir_outputs.push(Output { name: name.name, sr, params, annotated_type });
            if let Some(ts) = template_spec {
                template_specs.insert(sr, ts);
            }
            expressions.insert(sr, expression);
        }
        let hir_outputs = hir_outputs;
        //let mut hir_triggers = vec![];
        for (ix, t) in trigger.into_iter().enumerate() {
            let sr = SRef::OutRef(hir_outputs.len() + ix);
            let ast::Trigger { message, name, expression, .. } =
                Rc::try_unwrap(t).expect("other strong references should be dropped now");
            //hir_triggers.push(Trigger::new(name, message, sr));
            expressions.insert(sr, expression);
        }
        let hir_triggers = hir_triggers;
        let hir_inputs: Vec<Input> = inputs
            .into_iter()
            .enumerate()
            .map(|(ix, i)| Input {
                annotated_type: annotated_type(&i.ty).expect("Input Streams must have type annotation"),
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
            template_specs,
        };

        let next_input_ref = hir_inputs.len();
        let next_output_ref = hir_outputs.len() + hir_triggers.len();
        Hir { inputs: hir_inputs, outputs: hir_outputs, triggers: hir_triggers, next_input_ref, next_output_ref, mode }
        */
        todo!()
    }
}

impl Hir<Raw> {
    pub(crate) fn replace_expressions(self) -> Hir<IrExprMode> {
        unimplemented!()
    }
}
