use std::{
    borrow::Cow,
    io::{BufWriter, Write},
};

use dot::{LabelText, Style};

use super::{Mir, StreamAccessKind, StreamReference};

pub(super) struct DotRepresentation<'a> {
    mir: &'a Mir,
}

impl<'a> DotRepresentation<'a> {
    pub(super) fn compute_representation(mir: &'a Mir) -> String {
        let s = Self { mir };
        let res = Vec::new();
        let mut res_writer = BufWriter::new(res);
        s.write_dependency_graph(&mut res_writer);
        String::from_utf8(res_writer.into_inner().unwrap()).unwrap()
    }

    fn write_dependency_graph<W: Write>(self, output: &mut W) {
        dot::render(&self, output).unwrap()
    }
}

type DotGraphEdge = (StreamReference, StreamAccessKind, StreamReference);

impl<'a> dot::Labeller<'a, StreamReference, DotGraphEdge> for DotRepresentation<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("example1").unwrap()
    }

    fn node_id(&'a self, n: &StreamReference) -> dot::Id<'a> {
        let id = match n {
            StreamReference::In(i) => format!("In_{i}"),
            StreamReference::Out(i) => format!("Out_{i}"),
        };

        dot::Id::new(id).unwrap()
    }

    fn node_label<'b>(&'b self, n: &StreamReference) -> LabelText<'b> {
        let stream = self.mir.stream(*n);
        let stream_name = stream.name();
        let eval_layer: usize = stream.eval_layer().into();
        let memory_bound = stream.values_to_memorize().unwrap();
        let ty = stream.ty();

        let title_line = format!("<b>{stream_name}</b> : {ty}");
        let layer_line = format!("Layer: {eval_layer}");
        let memory_bound_line = format!("Memorization Bound: {memory_bound}");

        let label = format!(
            "{title_line}<br/>\
<font point-size=\"12\">{layer_line}<br/>
{memory_bound_line}</font>"
        );

        dot::LabelText::HtmlStr(label.into())
    }

    fn edge_label<'b>(&'b self, edge: &DotGraphEdge) -> LabelText<'b> {
        let kind = edge.1;
        let label = match kind {
            StreamAccessKind::Sync => "Sync".into(),
            StreamAccessKind::Hold => "Hold".into(),
            StreamAccessKind::Offset(o) => format!("Offset({o})"),
            StreamAccessKind::DiscreteWindow(_) => todo!(),
            StreamAccessKind::SlidingWindow(_) => todo!(),
            StreamAccessKind::Get => todo!(),
            StreamAccessKind::Fresh => todo!(),
        };
        dot::LabelText::LabelStr(label.into())
    }

    fn edge_style<'b>(&'b self, edge: &DotGraphEdge) -> Style {
        let kind = edge.1;
        match kind {
            StreamAccessKind::Hold => Style::Dashed,
            StreamAccessKind::Sync
            | StreamAccessKind::Offset(_)
            | StreamAccessKind::DiscreteWindow(_)
            | StreamAccessKind::SlidingWindow(_)
            | StreamAccessKind::Get
            | StreamAccessKind::Fresh => Style::None,
        }
    }

    fn node_shape<'b>(&'b self, node: &StreamReference) -> Option<LabelText<'b>> {
        let shape_str = match node {
            StreamReference::In(_) => "box",
            StreamReference::Out(_) => "ellipse",
        };

        Some(dot::LabelText::LabelStr(shape_str.into()))
    }
}

impl<'a> dot::GraphWalk<'a, StreamReference, DotGraphEdge> for DotRepresentation<'a> {
    fn nodes(&self) -> dot::Nodes<'a, StreamReference> {
        self.mir.all_streams().collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, DotGraphEdge> {
        let input_accesses = self
            .mir
            .inputs
            .iter()
            .map(|input| (input.reference, &input.accessed_by));
        let output_accesses = self
            .mir
            .outputs
            .iter()
            .map(|output| (output.reference, &output.accessed_by));
        let all_accesses = input_accesses.chain(output_accesses);

        let access_edges = all_accesses.flat_map(|(source_ref, accesses)| {
            accesses.iter().flat_map(move |(target_ref, access_kinds)| {
                access_kinds.iter().map(move |kind| (*target_ref, *kind, source_ref))
            })
        });

        // maybe we want to add other edges too
        let edges = access_edges.collect();

        Cow::Owned(edges)
    }

    fn source(&self, e: &DotGraphEdge) -> StreamReference {
        e.0
    }

    fn target(&self, e: &DotGraphEdge) -> StreamReference {
        e.2
    }
}

#[cfg(test)]
mod tests {
    use rtlola_parser::ParserConfig;

    #[test]
    fn test() {
        let spec = "input a : UInt64
        input b : UInt64
        output c := a + b.offset(by:-1).defaults(to:0)
        output d@a := b.hold().defaults(to:0)";

        let cfg = ParserConfig::for_string(String::from(spec));
        let mir = crate::parse(cfg).unwrap();

        println!("{}", mir.dot_representation());
    }
}
