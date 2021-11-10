use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::punctuated::{Pair, Punctuated};
use syn::{
    parse_macro_input, AttributeArgs, FnArg, Ident, ItemStruct, ItemTrait, Meta, NestedMeta, Pat, PatIdent, PatType,
    Path, Token, TraitItem, Type, Visibility,
};

#[proc_macro_derive(HirMode)]
pub fn derive_hir_mode(input: TokenStream) -> TokenStream {
    let s = parse_macro_input!(input as ItemStruct);
    let name = s.ident;
    let out = quote! {
        impl HirMode for #name {}
    };
    TokenStream::from(out)
}

#[proc_macro_attribute]
pub fn mode_functionality(_attr: TokenStream, input: TokenStream) -> TokenStream {
    let clone = input.clone();
    let s = parse_macro_input!(input as ItemTrait);
    let mut input = clone;
    let trait_name = s.ident;
    let inner_name = generate_inner_name(&trait_name);
    let wrapper_name = generate_wrapper_name(&trait_name);
    let inner_fn_name = generate_wrapper_fn_name(&trait_name);
    let wrapper = generate_wrapper(&trait_name, &wrapper_name, &inner_name, &inner_fn_name, &s.vis);
    let wrapper_impl = generate_wrapper_impl(&trait_name, &wrapper_name, &inner_name, &inner_fn_name);
    let blanket = generate_blanket(&trait_name, &wrapper_name, &inner_fn_name, &s.items);
    input.extend(wrapper);
    input.extend(wrapper_impl);
    input.extend(blanket);

    input
}

fn generate_inner_name(trait_name: &Ident) -> Ident {
    format_ident!("Inner{}", trait_name)
}

fn generate_wrapper_name(trait_name: &Ident) -> Ident {
    format_ident!("{}Wrapper", trait_name)
}

fn generate_wrapper_fn_name(trait_name: &Ident) -> Ident {
    format_ident!("inner_{}", trait_name)
}

fn generate_wrapper(
    trait_name: &Ident,
    wrapper_name: &Ident,
    inner_name: &Ident,
    inner_fn_name: &Ident,
    vis: &Visibility,
) -> TokenStream {
    let out = quote! {
        #vis trait #wrapper_name {
            type #inner_name: #trait_name;
            #[allow(non_snake_case)]
            fn #inner_fn_name(&self) -> &Self::#inner_name;
        }
    };
    TokenStream::from(out)
}

fn generate_wrapper_impl(
    trait_name: &Ident,
    wrapper_name: &Ident,
    inner_name: &Ident,
    inner_fn_name: &Ident,
) -> TokenStream {
    let out = quote! {
        impl<M> #wrapper_name for Hir<M>
        where
            M: HirMode + #trait_name
        {
            type #inner_name = M;
            fn #inner_fn_name(&self) -> &Self::#inner_name {
                &self.mode
            }
        }
    };
    TokenStream::from(out)
}

fn generate_blanket(
    trait_name: &Ident,
    wrapper_name: &Ident,
    inner_fn_name: &Ident,
    content: &[TraitItem],
) -> TokenStream {
    let content = content.iter().filter_map(|c| {
        match c {
            TraitItem::Method(m) => Some(m),
            _ => None,
        }
    });
    let sig = content.clone().map(|c| &c.sig);
    let args = content.clone().cloned().map(|c| c.sig.inputs).map(|args| {
        let mut ret = Punctuated::<Ident, Token![,]>::new();
        for (arg, opt_p) in args.into_pairs().map(Pair::into_tuple) {
            match arg {
                FnArg::Receiver(_) => {}, // Skip receiver (self etc)
                FnArg::Typed(PatType { pat, .. }) => {
                    match *pat {
                        Pat::Ident(PatIdent { ident, .. }) => {
                            ret.push_value(ident);
                            if let Some(p) = opt_p {
                                ret.push_punct(p);
                            }
                        },
                        _ => panic!("Inner WTF"),
                    }
                },
            }
        }
        ret
    });
    let fn_name = content.map(|c| &c.sig.ident);
    let out = quote! {
        impl<T> #trait_name for T
        where
            T: #wrapper_name,
        {
            #( #sig { self.#inner_fn_name().#fn_name(#args) })*
        }
    };
    TokenStream::from(out)
}

#[proc_macro_attribute]
pub fn covers_functionality(attr: TokenStream, input: TokenStream) -> TokenStream {
    let clone = input.clone();
    let s = parse_macro_input!(input as ItemStruct);
    let mode_name = s.ident;

    let attr = parse_macro_input!(attr as AttributeArgs);
    assert_eq!(attr.len(), 2);
    let sub_trait_name = extract_ident(&attr[0]);
    let accessor = extract_ident(&attr[1]);

    let sub_struct_name = s
        .fields
        .iter()
        .find_map(|field| {
            let name = field.ident.as_ref().expect("there can't be unnamed fields in structs");
            if accessor == name {
                if let Type::Path(tp) = &field.ty {
                    Some(tp.path.get_ident())
                } else {
                    panic!("type of accessor is not a path to a type")
                }
            } else {
                None
            }
        })
        .expect("accessor not found in struct");

    let sub_inner_name = generate_inner_name(sub_trait_name);
    let sub_wrapper_name = generate_wrapper_name(sub_trait_name);
    let sub_inner_fn_name = generate_wrapper_fn_name(sub_trait_name);

    let mut input = clone;
    let out = quote! {
        impl #sub_wrapper_name for #mode_name {
            type #sub_inner_name = #sub_struct_name;
            fn #sub_inner_fn_name(&self) -> &Self::#sub_inner_name {
                &self.#accessor
            }
        }
    };

    input.extend(TokenStream::from(out));

    input
}

fn extract_ident(nm: &NestedMeta) -> &Ident {
    extract_path(nm)
        .get_ident()
        .expect("extends_mode needs two arguments: the subsumed mode and a field refering to one.")
}

fn extract_path(nm: &NestedMeta) -> &Path {
    match nm {
        NestedMeta::Meta(Meta::Path(p)) => p,
        NestedMeta::Meta(_) | NestedMeta::Lit(_) => {
            panic!("extends_mode needs two arguments: the subsumed mode and a field refering to one.")
        },
    }
}
