use crate::hir::{Hir, Input, Output, Parameter};

//use crate::analysis::naming::{Declaration, DeclarationTable, NamingAnalysis};
use crate::ast;
use crate::ast::Constant;
use crate::common_ir::SRef;
use crate::parse::Span;
use crate::reporting::{Handler, LabeledSpan};
use crate::stdlib::FuncDecl;
use crate::FrontendConfig;
use crate::Raw;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Declaration {
    Const(Rc<Constant>),
    In(Rc<Input>),
    /// A non-parametric output
    Out(Rc<Output>),
    Func(Rc<FuncDecl>),
    Param(Rc<Parameter>), //TODO acutally use
                          //Type(Rc<ValueTy>),
}

impl Declaration {
    fn get_name(&self) -> Option<&str> {
        match self {
            Declaration::Const(constant) => Some(&constant.name.name),
            Declaration::In(input) => Some(&input.name),
            Declaration::Out(output) => Some(&output.name),
            Declaration::Param(p) => Some(&p.name),
            Declaration::Func(f) => Some(&f.name.name.name),
        }
    }
}

/// Provides a mapping from `String` to `Declaration` and is able to handle different scopes.
#[derive(Debug)]
pub(crate) struct ScopedDecl {
    scopes: Vec<HashMap<String, Declaration>>,
}

impl ScopedDecl {
    fn new() -> Self {
        ScopedDecl { scopes: vec![HashMap::new()] }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        assert!(self.scopes.len() > 1);
        self.scopes.pop();
    }

    fn get_decl_for(&self, name: &str) -> Option<Declaration> {
        for scope in self.scopes.iter().rev() {
            if let Some(decl) = scope.get(name) {
                return Some(decl.clone());
            }
        }
        None
    }

    fn get_decl_in_current_scope_for(&self, name: &str) -> Option<Declaration> {
        match self.scopes.last().expect("It appears that we popped the global context.").get(name) {
            Some(decl) => Some(decl.clone()),
            None => None,
        }
    }

    fn add_decl_for(&mut self, name: &str, decl: Declaration) {
        assert!(self.scopes.last().is_some());
        self.scopes.last_mut().expect("It appears that we popped the global context.").insert(name.to_string(), decl);
    }

    pub(crate) fn add_fun_decl(&mut self, fun: &FuncDecl) {
        assert!(self.scopes.last().is_some());
        let name = fun.name.to_string();
        self.scopes
            .last_mut()
            .expect("It appears that we popped the global context.")
            .insert(name, Declaration::Func(Rc::new(fun.clone())));
    }
}

pub(crate) struct NamingAnalysis<'a> {
    declarations: ScopedDecl,
    fun_declarations: ScopedDecl,
    expressions: HashMap<SRef, ast::Expression>,
    handler: &'a Handler,
}

pub(crate) const KEYWORDS: [&str; 26] = [
    "input",
    "output",
    "trigger",
    "import",
    "type",
    "self",
    "include",
    "invoke",
    "inv",
    "extend",
    "ext",
    "terminate",
    "ter",
    "unless",
    "if",
    "then",
    "else",
    "and",
    "or",
    "not",
    "forall",
    "exists",
    "any",
    "true",
    "false",
    "error",
];

impl<'a> NamingAnalysis<'a> {
    pub fn new(handler: &'a Handler, config: FrontendConfig) -> Self {
        let mut scoped_decls = ScopedDecl::new();

        // TODO config depended type declarations

        // add a new scope to distinguish between extern/builtin declarations
        scoped_decls.push();

        NamingAnalysis {
            declarations: ScopedDecl::new(),
            fun_declarations: ScopedDecl::new(),
            expressions: HashMap::new(),
            handler,
        }
    }

    pub fn check(&mut self, spec: &Hir<Raw>, constants: &Vec<Constant>) -> Option<HashMap<String, Declaration>> {
        crate::stdlib::import_implicit_module(&mut self.fun_declarations);
        //explicit imports
        /*
        for import in &spec.imports {
            match import.name.name.as_str() {
                "math" => stdlib::import_math_module(&mut self.fun_declarations),
                "regex" => stdlib::import_regex_module(&mut self.fun_declarations),
                n => self.handler.error_with_span(
                    &format!("unresolved import `{}`", n),
                    LabeledSpan::new(import.name.span, &format!("no `{}` in the root", n), true),
                ),
            }
        }
        */
        let mut result = HashMap::new();

        for constant in constants {
            let dec = Declaration::Const(Rc::new(constant.clone()));
            self.add_decl_for(dec.clone());
            result.insert(constant.name.name.clone(), dec);
        }
        for input in spec.inputs() {
            let dec = Declaration::In(Rc::new(input.clone()));
            self.add_decl_for(dec.clone());
            result.insert(input.name.clone(), dec);
        }
        for output in spec.outputs() {
            let dec = Declaration::Out(Rc::new(output.clone()));
            self.add_decl_for(dec.clone());
            result.insert(output.name.clone(), dec);
        }

        self.check_outputs(spec);
        if self.handler.contains_error() {
            Some(result)
        } else {
            None
        }
    }

    fn check_outputs(&mut self, spec: &Hir<Raw>) {
        for output in spec.outputs() {
            self.declarations.push();
            output.params.iter().for_each(|param| self.check_param(&Rc::new(param.clone())));
            //TODO TEMPLATE CHECK
            self.declarations.add_decl_for("self", Declaration::Out(Rc::new(output.clone())));
            let expr = &self.expressions[&output.sr].clone(); //TODO avoid clone, but its late
            self.check_expression(expr);
            self.declarations.pop();
        }
    }

    fn check_param(&mut self, param: &Rc<Parameter>) {
        if let Some(_) = self.declarations.get_decl_for(&param.name) {
            if let Some(_) = self.declarations.get_decl_in_current_scope_for(&param.name) {
                self.handler.error(&format!("re-definition of parameter `{}`", param.name));
                //FIXME
            }
        } else {
            self.add_decl_for(Declaration::Param(param.clone()));
        }
    }

    fn add_decl_for(&mut self, decl: Declaration) {
        let name = decl.get_name().expect("added declarations are guaranteed to have a name");
        //let span = decl.get_span().expect("all user defined declarations have a `Span`");
        // check for keyword
        let lower = name.to_lowercase();
        if KEYWORDS.contains(&lower.as_str()) {
            self.handler.error_with_span(
                &format!("`{}` is a reserved keyword", name),
                LabeledSpan::new(Span::unknown(), "use a different name here", true),
            )
        }

        if let Some(_decl) = self.declarations.get_decl_in_current_scope_for(name) {
            let mut builder = self.handler.build_error_with_span(
                &format!("the name `{}` is defined multiple times", name),
                LabeledSpan::new(Span::unknown(), &format!("`{}` redefined here", name), true),
            );
            builder.emit();
        } else {
            self.declarations.add_decl_for(name, decl.clone());
        }
    }

    fn check_expression(&mut self, expression: &ast::Expression) {
        use crate::ast::ExpressionKind::*;

        match &expression.kind {
            Ident(ident) => {
                if let None = self.declarations.get_decl_for(&ident.name) {
                    self.handler.error(&format!("unknown Identifier `{}` found", &ident.name))
                }
            }
            StreamAccess(expr, _) => self.check_expression(expr),
            Offset(expr, _) => {
                self.check_expression(expr);
            }
            SlidingWindowAggregation { expr, duration, .. } => {
                self.check_expression(expr);
                self.check_expression(duration);
            }
            DiscreteWindowAggregation { .. } => todo!(),
            Binary(_, left, right) => {
                self.check_expression(left);
                self.check_expression(right);
            }
            Lit(_) | MissingExpression => {}
            Ite(condition, if_case, else_case) => {
                self.check_expression(condition);
                self.check_expression(if_case);
                self.check_expression(else_case);
            }
            ParenthesizedExpression(_, expr, _) | Unary(_, expr) | Field(expr, _) => {
                self.check_expression(expr);
            }
            Tuple(exprs) => {
                exprs.iter().for_each(|expr| self.check_expression(expr));
            }
            Function(name, _, exprs) => {
                if let None = self.fun_declarations.get_decl_for(&name.to_string()) {
                    self.handler.error(&format!("unknown Function Name `{}` found", &name.to_string()))
                }
                exprs.iter().for_each(|expr| self.check_expression(expr));
            }
            Default(accessed, default) => {
                self.check_expression(accessed);
                self.check_expression(default);
            }
            Method(expr, _, _, args) => {
                self.check_expression(expr);
                args.iter().for_each(|expr| self.check_expression(expr));
            }
        }
    }
}
