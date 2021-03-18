use super::rusttyc::TcKey;
use crate::{common_ir::StreamReference, hir::Hir};
use crate::hir::expression::{ExprId, Expression};
use crate::hir::modes::HirMode;
use crate::hir::modes::IrExprTrait;
use crate::reporting::{Handler, Span};
use crate::tyc::pacing_types::ConcreteStreamPacing;
use crate::tyc::{
    pacing_ast_climber::PacingTypeChecker, pacing_types::ConcretePacingType, value_ast_climber::ValueTypeChecker,
    value_types::ConcreteValueType,
};
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LolaTypeChecker<'a, M>
where
    M: IrExprTrait + HirMode + 'static,
{
    pub(crate) hir: &'a Hir<M>,
    pub(crate) handler: &'a Handler,
    pub(crate) names: HashMap<StreamReference, &'a str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeId {
    SRef(StreamReference),
    Expr(ExprId),
    Param(usize, StreamReference),
}
pub(crate) trait Emittable {
    fn emit(
        self,
        handler: &Handler,
        spans: &[&HashMap<TcKey, Span>],
        names: &HashMap<StreamReference, &str>,
        key1: Option<TcKey>,
        key2: Option<TcKey>,
    );
}

pub(crate) struct TypeError<K: Emittable> {
    pub(crate) kind: K,
    pub(crate) key1: Option<TcKey>,
    pub(crate) key2: Option<TcKey>,
}

impl<E: Emittable> From<E> for TypeError<E> {
    fn from(kind: E) -> Self {
        TypeError { kind, key1: None, key2: None }
    }
}

impl<K: Emittable> TypeError<K> {
    pub(crate) fn emit(
        self,
        handler: &Handler,
        spans: &[&HashMap<TcKey, Span>],
        names: &HashMap<StreamReference, &str>,
    ) {
        self.kind.emit(handler, spans, names, self.key1, self.key2)
    }
}

#[derive(Debug, Clone)]
pub struct TypeTable {
    stream_types: HashMap<StreamReference, StreamType>,
    expression_types: HashMap<ExprId, StreamType>,
    param_types: HashMap<(StreamReference, usize), ConcreteValueType>,
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
    pub fn get_parameter_type(&self, stream: StreamReference, idx: usize) -> ConcreteValueType {
        self.param_types[&(stream, idx)].clone()
    }
}

#[derive(Debug, Clone)]
pub struct StreamType {
    pub value_ty: ConcreteValueType,
    pub pacing_ty: ConcretePacingType,
    pub spawn: (ConcretePacingType, Expression),
    pub filter: Expression,
    pub close: Expression,
}

impl StreamType {
    #[allow(dead_code)] // Todo: Actually use Typechecker
    pub fn get_value_type(&self) -> &ConcreteValueType {
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
    M: IrExprTrait + HirMode + 'static,
{
    pub fn new(hir: &'a Hir<M>, handler: &'a Handler) -> Self {
        let names: HashMap<StreamReference, &str> = hir
            .inputs()
            .map(|i| (i.sr, i.name.as_str()))
            .chain(hir.outputs().map(|o| (o.sr, o.name.as_str())))
            .collect();
        LolaTypeChecker { hir, handler, names }
    }

    pub fn check(&mut self) -> Result<TypeTable, String> {
        let pacing_tt = match self.pacing_type_infer() {
            Some(tt) => tt,
            None => return Err("Invalid Pacing Types".to_string()),
        };

        let value_tt = match self.value_type_infer(&pacing_tt) {
            Some(tt) => tt,
            None => return Err("Invalid Value Types".to_string()),
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
        let ptc = PacingTypeChecker::new(&self.hir, &self.names);
        ptc.type_check(self.handler)
    }

    pub(crate) fn value_type_infer(
        &self,
        pacing_tt: &HashMap<NodeId, ConcreteStreamPacing>,
    ) -> Option<HashMap<NodeId, ConcreteValueType>> {
        let mut ctx = ValueTypeChecker::new(&self.hir, pacing_tt);
        for input in self.hir.inputs() {
            if let Err(e) = ctx.input_infer(input) {
                e.emit(self.handler, &[&ctx.key_span], &self.names)
            }
        }

        for output in self.hir.outputs() {
            if let Err(e) = ctx.output_infer(output) {
                e.emit(self.handler, &[&ctx.key_span], &self.names)
            }
        }

        for trigger in self.hir.triggers() {
            if let Err(e) = ctx.trigger_infer(trigger) {
                e.emit(self.handler, &[&ctx.key_span], &self.names)
            }
        }

        if self.handler.contains_error() {
            return None;
        }

        let tt = match ctx.tyc.clone().type_check() {
            Ok(t) => t,
            Err(e) => {
                TypeError::from(e).emit(self.handler, &[&ctx.key_span], &self.names);
                return None;
            }
        };

        for err in ValueTypeChecker::<M>::check_explicit_bounds(ctx.annotated_checks.clone(), &tt) {
            err.emit(self.handler, &[&ctx.key_span], &self.names);
        }
        if self.handler.contains_error() {
            return None;
        }

        let result_map = ctx.node_key.into_iter().map(|(node, key)| (node, tt[&key].clone())).collect();
        Some(result_map)
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

#[cfg(test)]
mod tests {
    use crate::hir::modes::IrExprMode;
    use crate::parse::parse;
    use crate::reporting::Handler;
    use crate::tyc::rtltc::LolaTypeChecker;
    use crate::{RTLolaAst, RTLolaHIR};
    use std::path::PathBuf;

    fn setup_ast(spec: &str) -> (RTLolaHIR<IrExprMode>, Handler) {
        let handler = Handler::new(PathBuf::from("test"), spec.into());
        let ast: RTLolaAst = match parse(spec, &handler, crate::FrontendConfig::default()) {
            Ok(s) => s,
            Err(e) => panic!("Spec {} cannot be parsed: {}", spec, e),
        };
        let hir = crate::hir::RTLolaHIR::<IrExprMode>::from_ast(ast, &handler, &crate::FrontendConfig::default());
        (hir, handler)
    }

    #[test]
    fn type_table_creation() {
        let spec =  "input a: Int8\n input b: Int8\n output c(p) spawn with a := p + b\noutput d := c(b).hold().defaults(to: 0)\noutput e := c(d).hold().defaults(to: 0)";
        let (hir, handler) = setup_ast(spec);

        let mut tyc = LolaTypeChecker::new(&hir, &handler);
        tyc.check().unwrap();
    }
}
