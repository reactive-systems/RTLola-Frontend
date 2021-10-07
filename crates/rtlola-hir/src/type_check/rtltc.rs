use std::cmp::Ordering;
use std::collections::HashMap;

use rtlola_reporting::{Diagnostic, Handler, RtLolaError, Span};
use rusttyc::TcKey;

use crate::hir::{ExprId, Hir, StreamReference};
use crate::modes::{HirMode, Typed};
use crate::type_check::pacing_ast_climber::PacingTypeChecker;
use crate::type_check::value_ast_climber::ValueTypeChecker;
use crate::type_check::{ConcreteStreamPacing, ConcreteValueType, StreamType};

/// The [LolaTypeChecker] used to infer and check the RTLola type system for a given `Hir`.
#[derive(Clone, Debug)]
pub struct LolaTypeChecker<'a, M>
where
    M: HirMode,
{
    /// The [Hir] the checked is performed for.
    pub(crate) hir: &'a Hir<M>,
    /// The given [Handler] used for exact error reporting.
    pub(crate) handler: &'a Handler,
    /// A stream nme lookup table, generated for the input `Hir`.
    pub(crate) names: HashMap<StreamReference, &'a str>,
}

/// Wrapper enum to unify streams, expressions and parameter during inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeId {
    SRef(StreamReference),
    Expr(ExprId),
    Param(usize, StreamReference),
}

/// Resolvable is implemented for all type checker errors and is used for generic error printing.
pub(crate) trait Resolvable {
    fn into_diagnostic(
        self,
        spans: &[&HashMap<TcKey, Span>],
        names: &HashMap<StreamReference, &str>,
        key1: Option<TcKey>,
        key2: Option<TcKey>,
    ) -> Diagnostic;
}

pub(crate) struct TypeError<K: Resolvable> {
    pub(crate) kind: K,
    pub(crate) key1: Option<TcKey>,
    pub(crate) key2: Option<TcKey>,
}

impl<E: Resolvable> From<E> for TypeError<E> {
    fn from(kind: E) -> Self {
        TypeError {
            kind,
            key1: None,
            key2: None,
        }
    }
}

impl<K: Resolvable> TypeError<K> {
    pub(crate) fn into_diagnostic(
        self,
        spans: &[&HashMap<TcKey, Span>],
        names: &HashMap<StreamReference, &str>,
    ) -> Diagnostic {
        self.kind.into_diagnostic(spans, names, self.key1, self.key2)
    }
}

impl<'a, M> LolaTypeChecker<'a, M>
where
    M: HirMode + 'static,
{
    /// Constructs a new [LolaTypeChecker] given a `Hir`and `Handler`. Names table is constructed during call.
    pub(crate) fn new(hir: &'a Hir<M>, handler: &'a Handler) -> Self {
        let names: HashMap<StreamReference, &str> = hir
            .inputs()
            .map(|i| (i.sr, i.name.as_str()))
            .chain(hir.outputs().map(|o| (o.sr, o.name.as_str())))
            .collect();
        LolaTypeChecker { hir, handler, names }
    }

    /// Performs the complete type check procedure and a new HirMode or an error string.
    /// Detailed error information is emitted by the [Handler].
    pub(crate) fn check(&mut self) -> Result<Typed, RtLolaError> {
        let pacing_tt = self.pacing_type_infer()?;

        let value_tt = self.value_type_infer(&pacing_tt)?;

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
                },
                NodeId::Expr(id) => {
                    expression_map.insert(*id, st);
                },
                NodeId::Param(id, sref) => {
                    parameters.insert((*sref, *id), st.value_ty);
                },
            }
        });

        Ok(Typed::new(stream_map, expression_map, parameters))
    }

    /// starts the value type infer part with the [PacingTypeChecker].
    pub(crate) fn pacing_type_infer(&mut self) -> Result<HashMap<NodeId, ConcreteStreamPacing>, RtLolaError> {
        let ptc = PacingTypeChecker::new(&self.hir, &self.names);
        ptc.type_check()
    }

    /// starts the value type infer part with the [ValueTypeChecker].
    pub(crate) fn value_type_infer(
        &self,
        pacing_tt: &HashMap<NodeId, ConcreteStreamPacing>,
    ) -> Result<HashMap<NodeId, ConcreteValueType>, RtLolaError> {
        let ctx = ValueTypeChecker::new(&self.hir, &self.names, pacing_tt);
        ctx.type_check()
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
    use std::path::PathBuf;

    use rtlola_parser::ast::RtLolaAst;
    use rtlola_parser::{parse_with_handler, ParserConfig};
    use rtlola_reporting::Handler;

    use crate::hir::RtLolaHir;
    use crate::modes::BaseMode;
    use crate::type_check::rtltc::LolaTypeChecker;

    fn setup_ast(spec: &str) -> (RtLolaHir<BaseMode>, Handler) {
        let handler = Handler::new(PathBuf::from("test"), spec.into());
        let ast: RtLolaAst = match parse_with_handler(ParserConfig::for_string(spec.to_string()), &handler) {
            Ok(s) => s,
            Err(e) => panic!("Spec {} cannot be parsed: {}", spec, e),
        };
        let hir = crate::from_ast(ast, &handler).unwrap();
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
