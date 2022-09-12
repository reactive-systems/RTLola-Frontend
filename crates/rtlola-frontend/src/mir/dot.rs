use std::{
    borrow::Cow,
    collections::HashSet,
    io::{BufWriter, Write},
};

use dot::{LabelText, Style};
use uom::{si::frequency::hertz, fmt::DisplayStyle};

use super::{print::display_ac, ActivationCondition, Mir, StreamAccessKind, StreamReference, WindowReference};

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

#[derive(Clone, Copy, Debug)]
enum Node {
    Stream(StreamReference),
    Window(WindowReference),
}

impl From<StreamReference> for Node {
    fn from(s: StreamReference) -> Self {
        Node::Stream(s)
    }
}

impl From<WindowReference> for Node {
    fn from(w: WindowReference) -> Self {
        Node::Window(w)
    }
}

type Edge = (Node, EdgeType, Node);

#[derive(Clone, Debug)]
enum EdgeType {
    Access(StreamAccessKind),
    ActivationCondition,
}

impl From<StreamAccessKind> for EdgeType {
    fn from(ak: StreamAccessKind) -> Self {
        EdgeType::Access(ak)
    }
}

impl<'a> dot::Labeller<'a, Node, Edge> for DotRepresentation<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("example1").unwrap()
    }

    fn node_id(&'a self, n: &Node) -> dot::Id<'a> {
        let id = match n {
            Node::Stream(StreamReference::In(i)) => format!("In_{i}"),
            Node::Stream(StreamReference::Out(i)) => format!("Out_{i}"),
            Node::Window(WindowReference::Sliding(i)) => format!("W_{i}"),
            _ => unimplemented!(),
        };

        dot::Id::new(id).unwrap()
    }

    fn node_label<'b>(&'b self, n: &Node) -> LabelText<'b> {
        match n {
            Node::Stream(s) => self.build_stream_label(s),
            Node::Window(w) => self.build_window_label(w),
        }
    }

    fn edge_label<'b>(&'b self, edge: &Edge) -> LabelText<'b> {
        let kind = &edge.1;
        let label = match kind {
            EdgeType::Access(sak) => match sak {
                StreamAccessKind::Sync => "Sync".into(),
                StreamAccessKind::Hold => "Hold".into(),
                StreamAccessKind::Offset(o) => format!("Offset({o})"),
                StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => "".into(),
                StreamAccessKind::Get => todo!(),
                StreamAccessKind::Fresh => todo!(),
            },
            EdgeType::ActivationCondition => "".into(),
        };
        dot::LabelText::LabelStr(label.into())
    }

    fn edge_style<'b>(&'b self, edge: &Edge) -> Style {
        let kind = &edge.1;
        match kind {
            EdgeType::Access(StreamAccessKind::Hold) => Style::Dashed,
            EdgeType::ActivationCondition => Style::Dotted,
            EdgeType::Access(StreamAccessKind::Sync)
            | EdgeType::Access(StreamAccessKind::Offset(_))
            | EdgeType::Access(StreamAccessKind::DiscreteWindow(_))
            | EdgeType::Access(StreamAccessKind::SlidingWindow(_))
            | EdgeType::Access(StreamAccessKind::Get)
            | EdgeType::Access(StreamAccessKind::Fresh) => Style::None,
        }
    }

    fn node_shape<'b>(&'b self, node: &Node) -> Option<LabelText<'b>> {
        let shape_str = match node {
            Node::Stream(StreamReference::In(_)) => "box",
            Node::Stream(StreamReference::Out(_)) => "ellipse",
            Node::Window(_) => "note",
        };

        Some(dot::LabelText::LabelStr(shape_str.into()))
    }

    fn edge_end_arrow(&'a self, _e: &Edge) -> dot::Arrow {
        dot::Arrow::none()
    }

    fn edge_start_arrow(&'a self, _e: &Edge) -> dot::Arrow {
        dot::Arrow::normal()
    }
}

impl<'a> DotRepresentation<'a> {
    fn build_stream_label<'b>(&self, n: &StreamReference) -> LabelText<'b> {
        let stream = self.mir.stream(*n);
        let stream_name = stream.name();
        let eval_layer: usize = stream.eval_layer().into();
        let memory_bound = stream.values_to_memorize().unwrap();
        let ty = stream.ty();

        let title_line = format!("<b>{stream_name}</b> : {ty}");
        let layer_line = format!("Layer: {eval_layer}");
        let memory_bound_line = format!("Memorization Bound: {memory_bound}");

        match n {
            StreamReference::In(_) => Self::build_label(title_line, vec![layer_line, memory_bound_line]),
            StreamReference::Out(_) => {
                let pacing_ty = &self.mir.output(*n).eval.eval_pacing;
                let pacing_str = match pacing_ty {
                    super::PacingType::Periodic(f) => f.into_format_args(hertz, DisplayStyle::Abbreviation).to_string(),
                    super::PacingType::Event(ac) => display_ac(ac, &self.mir),
                    super::PacingType::Constant => todo!(),
                };
                let pacing_line = format!("Pacing: {pacing_str}");
                Self::build_label(title_line, vec![layer_line, memory_bound_line, pacing_line])
            },
        }
    }

    fn build_window_label<'b>(&self, n: &WindowReference) -> LabelText<'b> {
        let window = self.mir.sliding_window(*n);

        let title_line = match n {
            WindowReference::Sliding(w) => format!("Sliding Window {w}: {}", window.ty),
            WindowReference::Discrete(w) => format!("Discrete Window {w} : {}", window.ty),
        };

        let operation_line = format!("Window operation: {}", window.op);
        let duration_line = format!("Duration: {:?}", window.duration);
        Self::build_label(title_line, vec![operation_line, duration_line])
    }

    fn build_label(title_line: String, lines: Vec<String>) -> LabelText<'static> {
        let additional_lines = lines.join("<br/>");
        let label_text = if additional_lines.is_empty() {
            format!("{title_line}")
        } else {
            format!("{title_line}<br/><font point-size=\"12\">{additional_lines}</font>")
        };
        dot::LabelText::HtmlStr(label_text.into())
    }

    fn pacing_streams(ac: &ActivationCondition) -> HashSet<StreamReference> {
        match ac {
            ActivationCondition::Conjunction(c) | ActivationCondition::Disjunction(c) => c
                .iter()
                .map(|ac| Self::pacing_streams(ac))
                .reduce(|mut a, b| {
                    a.extend(b);
                    a
                })
                .unwrap(),
            ActivationCondition::Stream(s) => [*s].into(),
            ActivationCondition::True => HashSet::new(),
        }
    }
}

impl<'a> dot::GraphWalk<'a, Node, Edge> for DotRepresentation<'a> {
    fn nodes(&self) -> dot::Nodes<'a, Node> {
        let stream_nodes = self.mir.all_streams().map(|s| Node::Stream(s));
        let window_nodes = self.mir.sliding_windows.iter().map(|w| Node::Window(w.reference));

        stream_nodes.chain(window_nodes).collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge> {
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
                access_kinds.iter().filter_map(move |kind| match kind {
                    // we remove edges for window accesses, because we add extra nodes for them
                    StreamAccessKind::SlidingWindow(_) => None,
                    _ => Some((Node::Stream(source_ref), EdgeType::from(*kind), Node::Stream(*target_ref))),
                })
            })
        });

        let window_edges = self.mir.sliding_windows.iter().flat_map(|window| {
            [
                (
                    Node::Window(window.reference),
                    StreamAccessKind::SlidingWindow(window.reference).into(),
                    Node::Stream(window.caller),
                ),
                (
                    Node::Stream(window.target),
                    StreamAccessKind::SlidingWindow(window.reference).into(),
                    Node::Window(window.reference),
                ),
            ]
        });

        let ac_edges = self.mir.all_event_driven().into_iter().flat_map(|source_stream| {
            let ac = self.mir.get_ac(source_stream.reference).expect("all event driven");
            Self::pacing_streams(ac).into_iter().map(move |target_ref| {
                (
                    target_ref.into(),
                    EdgeType::ActivationCondition,
                    source_stream.reference.into(),
                )
            })
        });

        let edges = access_edges.chain(window_edges).chain(ac_edges).collect();

        Cow::Owned(edges)
    }

    fn source(&self, e: &Edge) -> Node {
        e.0
    }

    fn target(&self, e: &Edge) -> Node {
        e.2
    }
}

#[cfg(test)]
mod tests {
    use rtlola_parser::ParserConfig;

    #[test]
    fn test_dot() {
        let spec = "input a : UInt64
        input b : UInt64
        output c := a + b.offset(by:-1).defaults(to:0)
        output p@0.5Hz := b.aggregate(over:2s, using:sum)
        output d@a := b.hold().defaults(to:0)";

        let cfg = ParserConfig::for_string(String::from(spec));
        let mir = crate::parse(cfg).unwrap();

        println!("{}", mir.dot_representation());
    }
}
