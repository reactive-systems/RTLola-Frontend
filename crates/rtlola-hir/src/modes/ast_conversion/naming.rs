//! This module provides naming analysis for a given Lola AST.

use std::collections::HashMap;
use std::rc::Rc;

use rtlola_parser::ast::{Ident, NodeId, *};
use rtlola_reporting::{Diagnostic, RtLolaError, Span};

use crate::hir::AnnotatedType;
use crate::stdlib::FuncDecl;

/// Static vec of all Lola keywords, used to check for name conflicts. These MUST all be lowercase.
// TODO add an static assertion for this
pub(crate) const KEYWORDS: [&str; 24] = [
    "input", "output", "trigger", "import", "type", "self", "include", "spawn", "when", "close", "with", "unless",
    "if", "then", "else", "and", "or", "not", "forall", "exists", "any", "true", "false", "error",
];

/// The [DeclarationTable] maps a NodeId of an AST node to a [Declaration],
/// which holds a pointer to origin of the struct used at this AST Node.
pub(crate) type DeclarationTable = HashMap<NodeId, Declaration>;

/// [NamingAnalysis] performs a complete AST walk and checks for validity of all used identifiers.
/// # Procedure
/// Checks the following properties:
/// - Identifiers do not collide with keywords
/// - identifiers are unique in their matching scope
/// - used identifiers have valid definition
/// - all type annotations describe a valid type
#[derive(Debug)]
pub struct NamingAnalysis {
    declarations: ScopedDecl,
    type_declarations: ScopedDecl,
    fun_declarations: ScopedDecl,
    result: DeclarationTable,
}

impl NamingAnalysis {
    /// Constructs a new [NamingAnalysis] to perform checks on all identifiers. Use [check](NamingAnalysis::check) to start.
    pub fn new() -> Self {
        let mut scoped_decls = ScopedDecl::new();

        for (name, ty) in AnnotatedType::primitive_types() {
            scoped_decls.add_decl_for(name, Declaration::Type(Rc::new(ty.clone())));
        }

        // add a new scope to distinguish between extern/builtin declarations
        scoped_decls.push();

        NamingAnalysis {
            declarations: ScopedDecl::new(),
            type_declarations: scoped_decls,
            fun_declarations: ScopedDecl::new(),
            result: HashMap::new(),
        }
    }

    /// Adds a declaration to the declaration store.
    ///
    /// Checks if
    /// * name of declaration is a keyword
    /// * declaration already exists in current scope
    fn add_decl_for(&mut self, decl: Declaration) -> Result<(), RtLolaError> {
        assert!(!decl.is_type());
        let mut error = RtLolaError::new();
        let name = decl
            .get_name()
            .expect("added declarations are guaranteed to have a name");

        let span = decl.get_span().expect("all user defined declarations have a `Span`");

        // check for keyword
        let lower = name.to_lowercase();
        if KEYWORDS.contains(&lower.as_str()) {
            error.add(
                Diagnostic::error(&format!("`{name}` is a reserved keyword")).add_span_with_label(
                    span,
                    Some("use a different name here"),
                    true,
                ),
            );
        }

        let decl_name = if let Declaration::Func(rcfunc) = &decl {
            DeclName::Func(rcfunc.name.clone())
        } else {
            DeclName::Ident(name.to_string())
        };
        if let Some(decl) = self.declarations.get_decl_in_current_scope_for(&decl_name) {
            error.add(
                Diagnostic::error(&format!("the name `{name}` is defined multiple times"))
                    .add_span_with_label(span, Some(&format!("`{name}` redefined here")), true)
                    .maybe_add_span_with_label(
                        decl.get_span(),
                        Some(&format!("previous definition of the value `{name}` here")),
                        false,
                    ),
            );
        } else {
            self.declarations.add_decl_for(name, decl.clone());
        }
        Result::from(error)?;
        Ok(())
    }

    /// Checks if given type is bound
    fn check_type(&mut self, ty: &Type) -> Result<(), RtLolaError> {
        let mut error = RtLolaError::new();
        match &ty.kind {
            TypeKind::Simple(name) => {
                if let Some(decl) = self.type_declarations.get_decl_for(&DeclName::Ident(name.to_string())) {
                    assert!(decl.is_type());
                    self.result.insert(ty.id, decl);
                } else {
                    // it does not exist
                    error.add(
                        Diagnostic::error(&format!("cannot find type `{name}` in this scope")).add_span_with_label(
                            ty.span,
                            Some("not found in this scope"),
                            true,
                        ),
                    );
                }
            },
            TypeKind::Tuple(elements) => {
                elements.iter().for_each(|ty| {
                    if let Err(e) = self.check_type(ty) {
                        error.join(e);
                    }
                })
            },
            TypeKind::Optional(ty) => {
                if let Err(e) = self.check_type(ty) {
                    error.join(e);
                }
            },
        }
        Result::from(error)?;
        Ok(())
    }

    /// Checks that the parameter name and type are both valid
    fn check_param(&mut self, param: &Rc<Parameter>) -> Result<(), RtLolaError> {
        let mut error = RtLolaError::new();
        // check the name
        if let Some(decl) = self
            .declarations
            .get_decl_for(&DeclName::Ident(param.name.name.clone()))
        {
            assert!(!decl.is_type());

            // check if there is a parameter with the same name
            if let Some(decl) = self
                .declarations
                .get_decl_in_current_scope_for(&DeclName::Ident(param.name.name.clone()))
            {
                error.add(
                    Diagnostic::error(&format!(
                        "identifier `{}` is use more than once in this paramater list",
                        param.name.name
                    ))
                    .add_span_with_label(
                        param.name.span,
                        Some(&format!("`{}` used as a parameter more than once", param.name.name)),
                        true,
                    )
                    .add_span_with_label(
                        decl.get_span().expect("as it is in parameter list, it has a span"),
                        Some(&format!("previous use of the parameter `{}` here", param.name.name)),
                        false,
                    ),
                );
            }
        } else {
            // it does not exist
            if let Err(e) = self.add_decl_for(Declaration::Param(param.clone())) {
                error.join(e);
            }
        }

        // check the type is there exists a parameter type
        if let Some(param_ty) = param.ty.as_ref() {
            if let Err(e) = self.check_type(param_ty) {
                error.join(e);
            }
        }

        Result::from(error)
    }

    /// Entry method, checks that every identifier in the given spec is bound.
    pub(crate) fn check(&mut self, spec: &RtLolaAst) -> Result<DeclarationTable, RtLolaError> {
        use crate::stdlib;
        let mut error = RtLolaError::new();
        self.fun_declarations.add_all_fun_decl(stdlib::implicit_module());
        for import in &spec.imports {
            match import.name.name.as_str() {
                "math" => self.fun_declarations.add_all_fun_decl(stdlib::math_module()),
                "regex" => self.fun_declarations.add_all_fun_decl(stdlib::regex_module()),
                n => {
                    error.add(
                        Diagnostic::error(&format!("unresolved import `{n}`")).add_span_with_label(
                            import.name.span,
                            Some(&format!("no `{n}` in the root")),
                            true,
                        ),
                    );
                },
            }
        }

        // Store global declarations, i.e., constants, inputs, and outputs of the given specification
        for constant in &spec.constants {
            if let Err(e) = self.add_decl_for(Declaration::Const(constant.clone())) {
                error.join(e);
            }
            if let Some(ty) = constant.ty.as_ref() {
                if let Err(e) = self.check_type(ty) {
                    error.join(e);
                }
            }
        }

        for input in &spec.inputs {
            if let Err(e) = self.add_decl_for(Declaration::In(input.clone())) {
                error.join(e);
            }
            if let Err(e) = self.check_type(&input.ty) {
                error.join(e);
            }

            // check types for parametric inputs
            self.declarations.push();
            let err = input
                .params
                .iter()
                .flat_map(|param| self.check_param(param).err())
                .flatten()
                .collect();
            error.join(err);
            self.declarations.pop();
        }

        for output in &spec.outputs {
            if output.params.is_empty() {
                if let Err(e) = self.add_decl_for(Declaration::Out(output.clone())) {
                    error.join(e);
                }
            } else if let Err(e) = self.add_decl_for(Declaration::ParamOut(output.clone())) {
                error.join(e);
            }
            // Check annotated type if existing
            if let Some(output_ty) = output.annotated_type.as_ref() {
                if let Err(e) = self.check_type(output_ty) {
                    error.join(e);
                }
            }
        }

        if let Err(e) = self.check_outputs(spec) {
            error.join(e);
        }
        if let Err(e) = self.check_triggers(spec) {
            error.join(e);
        }

        Result::from(error)?;
        Ok(self.result.clone())
    }

    /// Checks that if the trigger has a name, it is unique
    fn check_triggers(&mut self, spec: &RtLolaAst) -> Result<(), RtLolaError> {
        let mut error = RtLolaError::new();
        for trigger in &spec.trigger {
            //Check that each supplied info stream exists
            for info_stream in &trigger.info_streams {
                if let Some(decl) = self
                    .declarations
                    .get_decl_for(&DeclName::Ident(info_stream.name.clone()))
                {
                    if !matches!(decl, Declaration::Out(_) | Declaration::In(_)) {
                        error.add(
                            Diagnostic::error("Only input and output names are supported in trigger messages.")
                                .add_span_with_label(info_stream.span, Some("Found other name here"), true),
                        );
                    }
                } else {
                    error.add(
                        Diagnostic::error(&format!("name `{}` does not exist in current scope", &info_stream.name))
                            .add_span_with_label(info_stream.span, Some("does not exist"), true),
                    );
                }
            }
            if let Some(pt) = trigger.annotated_pacing_type.as_ref() {
                if let Err(e) = self.check_expression(pt) {
                    error.join(e);
                }
            }
            self.declarations.push();
            if let Err(e) = self.check_expression(&trigger.expression) {
                error.join(e);
            }
            self.declarations.pop();
        }
        Result::from(error)
    }

    fn check_outputs(&mut self, spec: &RtLolaAst) -> Result<(), RtLolaError> {
        // recurse into expressions and check them
        let mut error = RtLolaError::new();
        for output in &spec.outputs {
            self.declarations.push();
            let para_errors = output
                .params
                .iter()
                .flat_map(|param| self.check_param(param).err())
                .flatten()
                .collect::<RtLolaError>();
            error.join(para_errors);
            if let Some(spawn) = &output.spawn {
                if let Some(spawn_expr) = &spawn.expression {
                    if let Err(e) = self.check_expression(spawn_expr) {
                        error.join(e);
                    }
                }
                if let Some(pacing) = &spawn.annotated_pacing {
                    if let Err(e) = self.check_expression(pacing) {
                        error.join(e);
                    }
                }
                if let Some(cond) = &spawn.condition {
                    if let Err(e) = self.check_expression(cond) {
                        error.join(e);
                    }
                }
            }
            if let Some(close) = &output.close {
                if let Err(e) = self.check_expression(&close.condition) {
                    error.join(e);
                }
                if let Some(pacing) = &close.annotated_pacing {
                    if let Err(e) = self.check_expression(pacing) {
                        error.join(e);
                    }
                }
            }

            for eval in &output.eval {
                if let Some(pt) = eval.annotated_pacing.as_ref() {
                    if let Err(e) = self.check_expression(pt) {
                        error.join(e);
                    }
                }
                if let Some(eval_cond) = &eval.condition {
                    if let Err(e) = self.check_expression(eval_cond) {
                        error.join(e);
                    }
                }
            }

            self.declarations.add_decl_for("self", Declaration::Out(output.clone()));
            for eval in &output.eval {
                if let Some(eval_expr) = &eval.eval_expression {
                    if let Err(e) = self.check_expression(eval_expr) {
                        error.join(e);
                    }
                }
            }
            self.declarations.pop();
        }
        error.into()
    }

    /// Checks that each used identifiers has a declaration in the current or higher scope.
    fn check_ident(&mut self, expression: &Expression, ident: &Ident) -> Result<(), Diagnostic> {
        if let Some(decl) = self.declarations.get_decl_for(&DeclName::Ident(ident.name.clone())) {
            assert!(!decl.is_type());
            self.result.insert(expression.id, decl);
            Ok(())
        } else {
            Err(
                Diagnostic::error(&format!("name `{}` does not exist in current scope", &ident.name))
                    .add_span_with_label(ident.span, Some("does not exist"), true),
            )
        }
    }

    /// Checks that each used function identifier has a declaration in the current scope or higher scope.
    fn check_function(&mut self, expression: &Expression, name: &FunctionName) -> Result<(), Diagnostic> {
        let str_repr = name.to_string();
        if let Some(decl) = self.fun_declarations.get_decl_for(&DeclName::Func(name.clone().into())) {
            assert!(decl.is_function());

            self.result.insert(expression.id, decl);
        } else if let Some(Declaration::ParamOut(out)) =
            self.declarations.get_decl_for(&DeclName::Ident(name.name.name.clone()))
        {
            // parametric outputs are represented as functions
            self.result.insert(expression.id, Declaration::ParamOut(out));
        } else {
            return Err(
                Diagnostic::error(&format!("function name `{str_repr}` does not exist in current scope"))
                    .add_span_with_label(name.name.span, Some("does not exist"), true),
            );
        }
        Ok(())
    }

    fn check_expression(&mut self, expression: &Expression) -> Result<(), RtLolaError> {
        use self::ExpressionKind::*;

        match &expression.kind {
            Ident(ident) => self.check_ident(expression, ident).map_err(RtLolaError::from),
            StreamAccess(expr, _) => self.check_expression(expr),
            Offset(expr, _) => self.check_expression(expr),
            DiscreteWindowAggregation { expr, duration, .. } => {
                RtLolaError::combine(self.check_expression(expr), self.check_expression(duration), |_, _| {})
            },
            SlidingWindowAggregation { expr, duration, .. } => {
                RtLolaError::combine(self.check_expression(expr), self.check_expression(duration), |_, _| {})
            },
            Binary(_, left, right) => {
                RtLolaError::combine(self.check_expression(left), self.check_expression(right), |_, _| {})
            },
            Lit(_) | MissingExpression => Ok(()),
            Ite(condition, if_case, else_case) => {
                let cond_errs: RtLolaError = self.check_expression(condition).into();
                let cons_errs: RtLolaError = self.check_expression(if_case).into();
                let alt_errs: RtLolaError = self.check_expression(else_case).into();
                cond_errs
                    .into_iter()
                    .chain(cons_errs.into_iter())
                    .chain(alt_errs.into_iter())
                    .collect::<RtLolaError>()
                    .into()
            },
            ParenthesizedExpression(_, expr, _) | Unary(_, expr) | Field(expr, _) => self.check_expression(expr),
            Tuple(exprs) => {
                exprs
                    .iter()
                    .flat_map(|expr| self.check_expression(expr).err())
                    .flatten()
                    .collect::<RtLolaError>()
                    .into()
            },
            Function(name, types, exprs) => {
                let func_err: RtLolaError = self.check_function(expression, name).map_err(RtLolaError::from).into();
                let type_errs: RtLolaError = types
                    .iter()
                    .flat_map(|ty| self.check_type(ty).err())
                    .flatten()
                    .collect();
                let expr_errs: RtLolaError = exprs
                    .iter()
                    .flat_map(|expr| self.check_expression(expr).err())
                    .flatten()
                    .collect();
                func_err
                    .into_iter()
                    .chain(type_errs.into_iter())
                    .chain(expr_errs.into_iter())
                    .collect::<RtLolaError>()
                    .into()
            },
            Default(accessed, default) => {
                RtLolaError::combine(
                    self.check_expression(accessed),
                    self.check_expression(default),
                    |_, _| {},
                )
            },
            Method(expr, name, types, args) => {
                // Method is equal to function with `expr` as first argument
                let func_name = FunctionName {
                    name: name.name.clone(),
                    arg_names: vec![None].into_iter().chain(name.arg_names.clone()).collect(),
                };
                let func_errs: RtLolaError = self
                    .check_function(expression, &func_name)
                    .map_err(RtLolaError::from)
                    .into();
                let type_errs: RtLolaError = types
                    .iter()
                    .flat_map(|ty| self.check_type(ty).err())
                    .flatten()
                    .collect();
                let expr_errs: RtLolaError = self.check_expression(expr).into();
                let arg_errs: RtLolaError = args
                    .iter()
                    .flat_map(|expr| self.check_expression(expr).err())
                    .flatten()
                    .collect();
                func_errs
                    .into_iter()
                    .chain(type_errs.into_iter())
                    .chain(expr_errs.into_iter())
                    .chain(arg_errs.into_iter())
                    .collect::<RtLolaError>()
                    .into()
            },
        }
    }
}

/// Provides a mapping from `String` to `Declaration` and is able to handle different scopes.
#[derive(Debug)]
pub(crate) struct ScopedDecl {
    scopes: Vec<HashMap<DeclName, Declaration>>,
}

impl ScopedDecl {
    fn new() -> Self {
        ScopedDecl {
            scopes: vec![HashMap::new()],
        }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        assert!(self.scopes.len() > 1);
        self.scopes.pop();
    }

    fn get_decl_for(&self, name: &DeclName) -> Option<Declaration> {
        for scope in self.scopes.iter().rev() {
            if let Some(decl) = scope.get(name) {
                return Some(decl.clone());
            }
        }
        None
    }

    fn get_decl_in_current_scope_for(&self, name: &DeclName) -> Option<Declaration> {
        self.scopes
            .last()
            .expect("It appears that we popped the global context.")
            .get(name)
            .cloned()
    }

    /// Adds a new declaration to the scope. Requires MANUEL check for duplicate definitions.
    fn add_decl_for(&mut self, name: &str, decl: Declaration) {
        assert!(self.scopes.last().is_some());
        self.scopes
            .last_mut()
            .expect("It appears that we popped the global context.")
            .insert(DeclName::Ident(name.to_string()), decl);
    }

    /// Adds a new function declaration to the scope. Requires MANUEL check for duplicate definitions.
    pub(crate) fn add_fun_decl(&mut self, fun: &FuncDecl) {
        assert!(self.scopes.last().is_some());
        self.scopes
            .last_mut()
            .expect("It appears that we popped the global context.")
            .insert(
                DeclName::Func(fun.name.clone()),
                Declaration::Func(Rc::new(fun.clone())),
            );
    }

    /// Adds all function declaration. See [add_fun_decl].
    pub(crate) fn add_all_fun_decl(&mut self, fns: Vec<&FuncDecl>) {
        fns.into_iter().for_each(|d| self.add_fun_decl(d));
    }
}

/// A [Declaration] unifies all possible elements which introduce new valid identifiers.
#[derive(Debug, Clone)]
pub(crate) enum Declaration {
    Const(Rc<Constant>),
    In(Rc<Input>),
    /// A non-parametric output
    Out(Rc<Output>),
    /// A paramertric output, internally represented as a function application
    ParamOut(Rc<Output>),
    Type(Rc<AnnotatedType>),
    Param(Rc<Parameter>),
    Func(Rc<FuncDecl>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DeclName {
    Func(crate::hir::FunctionName),
    Ident(String),
}

impl Declaration {
    fn get_span(&self) -> Option<Span> {
        match &self {
            Declaration::Const(constant) => Some(constant.name.span),
            Declaration::In(input) => Some(input.name.span),
            Declaration::Out(output) => Some(output.name.span),
            Declaration::ParamOut(output) => Some(output.name.span),
            Declaration::Param(p) => Some(p.name.span),
            Declaration::Type(_) | Declaration::Func(_) => None,
        }
    }

    fn get_name(&self) -> Option<&str> {
        match self {
            Declaration::Const(constant) => Some(&constant.name.name),
            Declaration::In(input) => Some(&input.name.name),
            Declaration::Out(output) => Some(&output.name.name),
            Declaration::ParamOut(output) => Some(&output.name.name),
            Declaration::Param(p) => Some(&p.name.name),
            Declaration::Type(_) | Declaration::Func(_) => None,
        }
    }

    fn is_type(&self) -> bool {
        match self {
            Declaration::Type(_) => true,
            Declaration::Const(_)
            | Declaration::In(_)
            | Declaration::Out(_)
            | Declaration::ParamOut(_)
            | Declaration::Param(_)
            | Declaration::Func(_) => false,
        }
    }

    fn is_function(&self) -> bool {
        matches!(self, Declaration::Func(_) | Declaration::ParamOut(_))
    }
}

impl From<FunctionName> for crate::hir::FunctionName {
    fn from(f: FunctionName) -> Self {
        crate::hir::FunctionName::new(
            f.name.name,
            &f.arg_names
                .iter()
                .map(|op| op.clone().map(|ident| ident.name))
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {

    use rtlola_parser::{parse, ParserConfig};

    use super::*;

    /// Parses the content, runs naming analysis, and returns number of errors
    fn number_of_naming_errors(content: &str) -> usize {
        let ast = parse(ParserConfig::for_string(content.to_string())).unwrap_or_else(|e| panic!("{:?}", e));
        let mut naming_analyzer = NamingAnalysis::new();
        match naming_analyzer.check(&ast) {
            Ok(_) => 0,
            Err(e) => e.num_errors(),
        }
    }

    #[test]
    fn unknown_types_are_reported() {
        assert_eq!(3, number_of_naming_errors("output test(ab: B, c: D): E := 3"))
    }

    #[test]
    fn unknown_identifiers_are_reported() {
        assert_eq!(1, number_of_naming_errors("output test: Int8 := A"))
    }

    #[test]
    fn primitive_types_are_a_known() {
        for ty in &[
            "Int8", "Int16", "Int32", "Int64", "Float32", "Float64", "Bool", "String",
        ] {
            assert_eq!(0, number_of_naming_errors(&format!("output test: {} := 3", ty)))
        }
    }

    #[test]
    fn duplicate_names_at_the_same_level_are_reported() {
        assert_eq!(
            1,
            number_of_naming_errors("output test: String := 3\noutput test: String := 3")
        )
    }

    #[test]
    fn duplicate_parameters_are_not_allowed_for_outputs() {
        assert_eq!(1, number_of_naming_errors("output test(ab: Int8, ab: Int8) := 3"))
    }

    #[test]
    fn duplicate_parameters_are_not_allowed_for_inputs() {
        assert_eq!(1, number_of_naming_errors("input test(ab: Int8, ab: Int8) : Int8"))
    }

    #[test]
    fn keyword_are_not_valid_names() {
        assert_eq!(1, number_of_naming_errors("output if := 3"))
    }

    #[test]
    fn template_spec_is_also_tested() {
        assert_eq!(1, number_of_naming_errors("output a spawn with b eval with 3"))
    }

    #[test]
    fn self_is_allowed_in_output_expression() {
        assert_eq!(0, number_of_naming_errors("output a  := self[-1]"))
    }

    #[test]
    fn self_not_allowed_in_trigger_expression() {
        //assert_eq!(1, number_of_naming_errors("trigger a  := self[-1]"))
        assert_eq!(1, number_of_naming_errors("trigger self[-1]"))
    }

    #[test]
    fn unknown_import() {
        assert_eq!(1, number_of_naming_errors("import xzy"))
    }

    #[test]
    fn known_import() {
        assert_eq!(0, number_of_naming_errors("import math"))
    }

    #[test]
    fn unknown_function() {
        assert_eq!(1, number_of_naming_errors("output x: Float32 := sqrt(2)"))
    }

    #[test]
    fn wrong_arity_function() {
        let spec = "import math\noutput o: Float64 := cos()";
        assert_eq!(1, number_of_naming_errors(spec));
    }

    #[test]
    fn known_function_though_import() {
        assert_eq!(0, number_of_naming_errors("import math\noutput x: Float32 := sqrt(2)"))
    }

    #[test]
    fn missing_expression() {
        // should not produce an error as we want to be able to handle incomplete specs in analysis
        assert_eq!(
            0,
            number_of_naming_errors("input x: Bool\noutput y: Bool := \ntrigger (y || x)")
        )
    }

    #[test]
    fn parametric_output() {
        let spec = "output x (a: UInt8, b: Bool): Int8 := 1 output y := x(1, false)";
        assert_eq!(0, number_of_naming_errors(spec));
    }

    #[test]
    fn test_param_spec_wrong_parameters() {
        let spec = "input in(a: Int8, b: Int8): Int8\noutput x := in(1)";
        assert_eq!(1, number_of_naming_errors(spec));
    }

    #[test]
    fn simple_variable_use() {
        assert_eq!(number_of_naming_errors("output a: Int8 := 3 output b: Int32 := a"), 0)
    }

    #[test]
    fn test_aggregate() {
        let spec = "output a @1Hz := 1 output b @1min:= a.aggregate(over: 1s, using: sum)";
        assert_eq!(0, number_of_naming_errors(spec));
    }

    #[test]
    fn test_param_use() {
        let spec = "output a(x,y,z) := if y then y else z";
        assert_eq!(0, number_of_naming_errors(spec));
    }

    #[test]
    fn test_trigger_infos() {
        let spec = "input a: Int8\ntrigger a \"test msg\" (a)";
        assert_eq!(0, number_of_naming_errors(spec));
    }

    #[test]
    fn test_trigger_infos_fail() {
        let spec = "input a: Int8\ntrigger a \"test msg\" (a, b)";
        assert_eq!(1, number_of_naming_errors(spec));
    }

    #[test]
    fn test_trigger_infos_fail2() {
        let spec = "input a: Int8\noutput b (x:Int8) := 42\ntrigger a \"test msg\" (a, b)";
        assert_eq!(1, number_of_naming_errors(spec));
    }
}
