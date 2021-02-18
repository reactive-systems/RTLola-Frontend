use super::*;

use crate::common_ir::StreamReference;
use crate::hir::expression::{Constant, ConstantLiteral, ExprId, Expression, ExpressionKind};
use crate::hir::modes::ir_expr::WithIrExpr;
use crate::hir::modes::HirMode;
use crate::reporting::{Handler, Span};
use crate::tyc::pacing_types::{AbstractExpressionType, ConcreteStreamPacing, PacingError};
use crate::tyc::{
    pacing_ast_climber::Context as PacingContext, pacing_types::ConcretePacingType, value_ast_climber::ValueContext,
    value_types::IConcreteType,
};
use crate::RTLolaHIR;
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
    #[allow(dead_code)] // Todo: Actually use Typetable
    /// For a given StreamReference, lookup the corresponding StreamType.
    pub fn get_type_for_stream(&self, sref: StreamReference) -> StreamType {
        self.stream_types[&sref].clone()
    }

    #[allow(dead_code)] // Todo: Actually use Typetable
    /// For a given Expression Id, lookup the corresponding StreamType.
    pub fn get_type_for_expr(&self, exprid: ExprId) -> StreamType {
        self.expression_types[&exprid].clone()
    }

    #[allow(dead_code)] // Todo: Actually use Typetable
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
    #[allow(dead_code)] // Todo: Actually use Typechecker
    pub fn get_value_type(&self) -> &IConcreteType {
        &self.value_ty
    }

    #[allow(dead_code)] // Todo: Actually use Typechecker
    pub fn get_pacing_type(&self) -> &ConcretePacingType {
        &self.pacing_ty
    }

    #[allow(dead_code)] // Todo: Actually use Typechecker
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

        let value_tt = match self.value_type_infer(&pacing_tt) {
            Ok(tt) => tt,
            Err(e) => return Err(e),
        };

        let mut expression_map = HashMap::new();
        let mut stream_map = HashMap::new();
        let mut parameters = HashMap::new();
        value_tt.keys().for_each(|id| {
            let concrete_pacing = pacing_tt[id].clone();
            let st = StreamType {
                value_ty: value_tt[id].clone(),
                pacing_ty: concrete_pacing.expression_pacing,
                filter: concrete_pacing.filter,
                spawn: (concrete_pacing.spawn.0, concrete_pacing.spawn.1),
                close: concrete_pacing.close,
            };
            // Todo: Upto here
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

        Ok(TypeTable { stream_types: stream_map, expression_types: expression_map, param_types: parameters })
    }

    pub(crate) fn pacing_type_infer(&mut self) -> Option<HashMap<NodeId, ConcreteStreamPacing>> {
        let stream_names: HashMap<StreamReference, &str> = self
            .hir
            .inputs()
            .map(|i| (i.sr, i.name.as_str()))
            .chain(self.hir.outputs().map(|o| (o.sr, o.name.as_str())))
            .collect();
        let mut ctx = PacingContext::new(&self.hir);
        for input in self.hir.inputs() {
            if let Err(e) = ctx.input_infer(input) {
                e.emit(self.handler, &ctx.pacing_key_span, &ctx.expression_key_span, &stream_names);
            }
        }

        for output in self.hir.outputs() {
            if let Err(e) = ctx.output_infer(output) {
                e.emit(self.handler, &ctx.pacing_key_span, &ctx.expression_key_span, &stream_names);
            }
        }

        for trigger in self.hir.triggers() {
            if let Err(e) = ctx.trigger_infer(trigger) {
                e.emit(self.handler, &ctx.pacing_key_span, &ctx.expression_key_span, &stream_names);
            }
        }

        let nid_key = ctx.node_key.clone();
        let pacing_tt = match ctx.pacing_tyc.type_check() {
            Ok(t) => t,
            Err(e) => {
                PacingError::from(e).emit(self.handler, &ctx.pacing_key_span, &ctx.expression_key_span, &stream_names);
                return None;
            }
        };
        let exp_tt = match ctx.expression_tyc.type_check() {
            Ok(t) => t,
            Err(e) => {
                PacingError::from(e).emit(self.handler, &ctx.pacing_key_span, &ctx.expression_key_span, &stream_names);
                return None;
            }
        };

        if self.handler.contains_error() {
            return None;
        }

        for pe in PacingContext::post_process(&self.hir, nid_key, &pacing_tt, &exp_tt) {
            pe.emit(self.handler, &ctx.pacing_key_span, &ctx.expression_key_span, &stream_names);
        }
        if self.handler.contains_error() {
            return None;
        }

        let pacing_key_span = ctx.pacing_key_span.clone();
        let exp_key_span = ctx.expression_key_span.clone();

        let exp_top = Expression {
            kind: ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(true))),
            eid: ExprId(u32::max_value()),
            span: Span::Unknown,
        };

        let exp_bot = Expression {
            kind: ExpressionKind::LoadConstant(Constant::BasicConstant(ConstantLiteral::Bool(false))),
            eid: ExprId(u32::max_value()),
            span: Span::Unknown,
        };

        let ctt: HashMap<NodeId, ConcreteStreamPacing> = ctx
            .node_key
            .iter()
            .filter_map(|(id, key)| {
                let exp_pacing = match ConcretePacingType::from_abstract(pacing_tt[key.exp_pacing].clone()) {
                    Ok(ct) => ct,
                    Err(e) => {
                        e.emit(self.handler, &pacing_key_span, &exp_key_span, &stream_names);
                        return None;
                    }
                };
                let spawn_pacing = match ConcretePacingType::from_abstract(pacing_tt[key.spawn.0].clone()) {
                    Ok(ct) => ct,
                    Err(e) => {
                        e.emit(self.handler, &pacing_key_span, &exp_key_span, &stream_names);
                        return None;
                    }
                };
                let spawn_condition_expression = match &exp_tt[key.spawn.1] {
                    AbstractExpressionType::Any => exp_top.clone(),
                    AbstractExpressionType::Expression(e) => e.clone(),
                };

                let filter = match &exp_tt[key.filter] {
                    AbstractExpressionType::Any => exp_top.clone(),
                    AbstractExpressionType::Expression(e) => e.clone(),
                };

                let close = match &exp_tt[key.close] {
                    AbstractExpressionType::Any => exp_bot.clone(),
                    AbstractExpressionType::Expression(e) => e.clone(),
                };

                Some((
                    *id,
                    ConcreteStreamPacing {
                        expression_pacing: exp_pacing,
                        spawn: (spawn_pacing, spawn_condition_expression),
                        filter,
                        close,
                    },
                ))
            })
            .collect();

        if self.handler.contains_error() {
            return None;
        }
        Some(ctt)
    }

    pub(crate) fn value_type_infer(
        &self,
        pacing_tt: &HashMap<NodeId, ConcreteStreamPacing>,
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
        dbg!(&tt);
        /*
        let bm = ctx.node_key;
        for (nid, k) in bm.iter() {
            //DEBUGG
            println!("{:?}", (*nid, tt[*k].clone()));
        }
        */
        let rtt_r = tt.try_reified();
        if rtt_r.is_err() {
            return Err(format!("Typetable not reifiable: {:?}", rtt_r.unwrap_err()));
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
