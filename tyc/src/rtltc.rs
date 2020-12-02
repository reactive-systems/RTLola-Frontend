use super::*;

use crate::pacing_ast_climber::Context as PacingContext;
use crate::pacing_types::{emit_error, ConcretePacingType};
use crate::value_ast_climber::ValueContext;
use crate::value_types::IConcreteType;
use front::common_ir::StreamReference;
use front::hir::expression::{ExprId, Expression};
use front::hir::modes::ir_expr::WithIrExpr;
use front::hir::modes::HirMode;
use front::reporting::Handler;
use front::RTLolaHIR;
use rusttyc::types::ReifiedTypeTable;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LolaTypeChecker<'a, M>
where
    M: WithIrExpr + HirMode + 'static,
{
    pub(crate) hir: &'a RTLolaHIR<M>,
    pub(crate) handler: &'a Handler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeId {
    SRef(StreamReference),
    Expr(ExprId),
    Param(usize, StreamReference),
}

#[derive(Debug, Clone)]
pub struct TypeTable {
    stream_types: HashMap<StreamReference, StreamType>,
    expression_types: HashMap<ExprId, StreamType>,
    param_types: HashMap<(StreamReference, usize), IConcreteType>,
}

impl TypeTable {
    /// For a given StreamReference, lookup the corresponding StreamType.
    pub fn get_type_for_stream(&self, sref: StreamReference) -> StreamType {
        self.stream_types[&sref].clone()
    }

    /// For a given Expression Id, lookup the corresponding StreamType.
    pub fn get_type_for_expr(&self, exprid: ExprId) -> StreamType {
        self.expression_types[&exprid].clone()
    }

    /// Returns the Value Type of the `idx`-th Parameter for the Stream `stream`.
    pub fn get_parameter_type(&self, stream: StreamReference, idx: usize) -> IConcreteType {
        self.param_types[&(stream, idx)].clone()
    }
}

#[derive(Debug, Clone)]
pub struct StreamType {
    pub value_ty: IConcreteType,
    pub pacing_ty: ConcretePacingType,
    pub spawn: (ConcretePacingType, Expression),
    pub filter: Expression,
    pub close: Expression,
}

impl StreamType {
    pub fn get_value_type(&self) -> &IConcreteType {
        &self.value_ty
    }

    pub fn get_pacing_type(&self) -> &ConcretePacingType {
        &self.pacing_ty
    }

    pub fn get_instance_expressions(&self) -> (&Expression, &Expression, &Expression) {
        (&self.spawn.1, &self.filter, &self.close)
    }
}

impl<'a, M> LolaTypeChecker<'a, M>
where
    M: WithIrExpr + HirMode + 'static,
{
    pub fn new(hir: &'a RTLolaHIR<M>, handler: &'a Handler) -> Self {
        LolaTypeChecker { hir, handler }
    }

    pub fn check(&mut self) -> Result<TypeTable, String> {
        let pacing_tt = match self.pacing_type_infer() {
            Some(tt) => tt,
            None => return Err("Invalid Pacing Types".to_string()),
        };
        //TODO imports
        let value_tt = match self.value_type_infer(&pacing_tt) {
            Ok(tt) => tt,
            Err(e) => return Err(e),
        };

        //Spawn type:
        //TODO

        //Filter type:
        //TODO

        //Close Type:
        //TODO

        let mut expression_map = HashMap::new();
        let mut stream_map = HashMap::new();
        let mut parameters = HashMap::new();
        value_tt.keys().for_each(|id| {
            let st = StreamType {
                value_ty: value_tt[id].clone(),
                pacing_ty: pacing_tt[id].clone(),
                filter: todo!(),
                spawn: todo!(),
                close: todo!(),
            };
            match id {
                NodeId::SRef(sref) => {
                    stream_map.insert(*sref, st);
                }
                NodeId::Expr(id) => {
                    expression_map.insert(*id, st);
                }
                NodeId::Param(id, sref) => {
                    parameters.insert((*sref, *id), st.value_ty);
                }
            }
        });

        Ok(TypeTable {
            stream_types: stream_map,
            expression_types: expression_map,
            param_types: parameters,
        })
    }

    pub(crate) fn pacing_type_infer(&mut self) -> Option<HashMap<NodeId, ConcretePacingType>> {
        let mut ctx = PacingContext::new(&self.hir);
        let stream_names: HashMap<StreamReference, &str> = self
            .hir
            .inputs()
            .map(|i| (i.sr, i.name.as_str()))
            .chain(self.hir.outputs().map(|o| (o.sr, o.name.as_str())))
            .collect();
        for input in self.hir.inputs() {
            if let Err(e) = ctx.input_infer(input) {
                emit_error(&e, self.handler, &ctx.key_span, &stream_names);
            }
        }

        for output in self.hir.outputs() {
            if let Err(e) = ctx.output_infer(output) {
                emit_error(&e, self.handler, &ctx.key_span, &stream_names);
            }
        }

        for trigger in self.hir.triggers() {
            if let Err(e) = ctx.trigger_infer(trigger) {
                emit_error(&e, self.handler, &ctx.key_span, &stream_names);
            }
        }

        let nid_key = ctx.node_key.clone();
        print!("{:?}", &ctx.tyc);
        let tt = match ctx.tyc.type_check() {
            Ok(t) => t,
            Err(e) => {
                emit_error(&e, self.handler, &ctx.key_span, &stream_names);
                return None;
            }
        };

        if self.handler.contains_error() {
            return None;
        }

        for pe in PacingContext::post_process(&self.hir, nid_key, &tt) {
            pe.emit(self.handler, &ctx.key_span, &stream_names, None, None);
        }
        if self.handler.contains_error() {
            return None;
        }

        let key_span = ctx.key_span.clone();
        let ctt: HashMap<NodeId, ConcretePacingType> = ctx
            .node_key
            .iter()
            .filter_map(
                |(id, key)| match ConcretePacingType::from_abstract(tt[*key].clone()) {
                    Ok(ct) => Some((*id, ct)),
                    Err(e) => {
                        e.emit(self.handler, &key_span, &stream_names, None, None);
                        None
                    }
                },
            )
            .collect();

        if self.handler.contains_error() {
            return None;
        }
        Some(ctt)
    }

    pub(crate) fn value_type_infer(
        &self,
        pacing_tt: &HashMap<NodeId, ConcretePacingType>,
    ) -> Result<HashMap<NodeId, IConcreteType>, String> {
        //let value_tyc = rusttyc::TypeChecker::new();

        let mut ctx = ValueContext::new(&self.hir, self.handler, pacing_tt);

        for input in self.hir.inputs() {
            if let Err(e) = ctx.input_infer(input) {
                let msg = ctx.handle_error(e);
                return Err(msg);
            }
        }

        for output in self.hir.outputs() {
            if let Err(e) = ctx.output_infer(output) {
                let msg = ctx.handle_error(e);
                return Err(msg);
            }
        }

        for trigger in self.hir.triggers() {
            if let Err(e) = ctx.trigger_infer(trigger) {
                let msg = ctx.handle_error(e);
                return Err(msg);
            }
        }

        let tt_r = ctx.tyc.clone().type_check();
        if let Err(tc_err) = tt_r {
            let msg: String = ctx.handle_error(tc_err);
            return Err(msg);
        }
        let tt = tt_r.expect("Ensured by previous cases");
        /*
        let bm = ctx.node_key;
        for (nid, k) in bm.iter() {
            //DEBUGG
            println!("{:?}", (*nid, tt[*k].clone()));
        }
        */
        let rtt_r = tt.try_reified();
        if rtt_r.is_err() {
            return Err("TypeTable not reifiable: ValueType not constrained enough".to_string());
        }
        let rtt: ReifiedTypeTable<IConcreteType> = rtt_r.unwrap();
        let mut result_map = HashMap::new();
        for (nid, k) in ctx.node_key.iter() {
            result_map.insert(*nid, rtt[*k].clone());
        }
        Ok(result_map)
    }
}

impl PartialOrd for NodeId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (NodeId::Expr(a), NodeId::Expr(b)) => Some(a.cmp(&b)),
            (NodeId::SRef(a), NodeId::SRef(b)) => Some(a.cmp(&b)),
            (NodeId::Param(_, _), _) => unreachable!(),
            (_, _) => None,
        }
    }
}
