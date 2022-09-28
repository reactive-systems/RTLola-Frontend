use std::{borrow::Cow, collections::HashMap, io::BufWriter};

use dot::{LabelText, Style};
use serde::Serialize;

use super::{print::display_pacing_type, Mir, StreamAccessKind, StreamReference, TriggerReference, WindowReference};

#[derive(Debug, Clone)]
pub struct DependencyGraph<'a> {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    infos: HashMap<Node, NodeInformation<'a>>,
}

impl<'a> DependencyGraph<'a> {
    pub(super) fn new(mir: &'a Mir) -> Self {
        let stream_nodes = mir.all_streams().map(|s| Node::Stream(s));
        let window_nodes = mir.sliding_windows.iter().map(|w| Node::Window(w.reference));
        let trigger_nodes = mir.triggers.iter().map(|trigger| Node::Trigger(trigger.trigger_reference));
        let nodes: Vec<_> = stream_nodes.chain(window_nodes).chain(trigger_nodes).collect();

        let edges = edges(mir);

        let infos = nodes.iter().map(|node| (*node, node_infos(mir, *node))).collect();

        Self { nodes, edges, infos }
    }

    pub fn dot(&self) -> String {
        let res = Vec::new();
        let mut res_writer = BufWriter::new(res);
        dot::render(self, &mut res_writer).unwrap();
        String::from_utf8(res_writer.into_inner().unwrap()).unwrap()
    }

    pub fn json(&self) -> String {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Node {
    Stream(StreamReference),
    Trigger(TriggerReference),
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
}

impl From<StreamAccessKind> for EdgeType {
    fn from(ak: StreamAccessKind) -> Self {
        EdgeType::Access(ak)
    }
}

#[derive(Serialize, Debug, Clone)]
enum NodeInformation<'a> {
    Stream {
        kind: StreamKind,
        stream_name: &'a str,
        eval_layer: usize,
        memory_bound: u32,
        pacing_ty: String,
        value_ty: String,
    },

    Trigger {
        idx: usize,
        eval_layer: usize,
        pacing_ty: String,
        value_ty: String,
        message: &'a str,
    },

    Window,
}

#[derive(Serialize, Clone, Debug)]
enum StreamKind {
    Input,
    Output,
}

fn node_infos(mir: &Mir, node: Node) -> NodeInformation {
    match node {
        Node::Stream(sref) => stream_infos(mir, sref),
        Node::Window(wref) => window_infos(mir, wref),
        Node::Trigger(tref) => trigger_infos(mir, tref),
    }
}

fn stream_infos(mir: &Mir, sref: StreamReference) -> NodeInformation {
    let stream = mir.stream(sref);

    let stream_name = stream.name();
    let eval_layer: usize = stream.eval_layer().into();
    let memory_bound = stream.values_to_memorize().unwrap();
    let value_ty = stream.ty();
    let value_str = value_ty.to_string();

    let pacing_str = if let StreamReference::Out(_) = sref {
        let pacing_ty = &mir.output(sref).eval.eval_pacing;
        display_pacing_type(mir, pacing_ty)
    } else {
        "input".into()
    };

    let kind = if stream.is_input() {
        StreamKind::Input
    } else {
        StreamKind::Output
    };

    NodeInformation::Stream {
        kind,
        stream_name,
        eval_layer,
        pacing_ty: pacing_str,
        value_ty: value_str,
        memory_bound,
    }
}

fn window_infos(mir: &Mir, window: WindowReference) -> NodeInformation {
    NodeInformation::Window
}

fn trigger_infos(mir: &Mir, tref: TriggerReference) -> NodeInformation {
    let trigger = &mir.triggers[tref];
    if let NodeInformation::Stream {
        kind: _,
        stream_name: _,
        eval_layer,
        memory_bound: _,
        pacing_ty,
        value_ty,
    } = stream_infos(mir, trigger.reference)
    {
        NodeInformation::Trigger {
            idx: tref,
            eval_layer: eval_layer,
            pacing_ty: pacing_ty,
            value_ty: value_ty,
            message: &trigger.message,
        }
    } else {
        unreachable!("is NodeInformation::Stream");
    }
}

fn edges(mir: &Mir) -> Vec<Edge> {
    let input_accesses = mir.inputs.iter().map(|input| (input.reference, &input.accessed_by));
    let output_accesses = mir.outputs.iter().map(|output| (output.reference, &output.accessed_by));
    let all_accesses = input_accesses.chain(output_accesses);

    let access_edges = all_accesses.flat_map(|(source_ref, accesses)| {
        accesses.iter().flat_map(move |(target_ref, access_kinds)| {
            access_kinds.iter().filter_map(move |kind| match kind {
                // we remove edges for window accesses, because we add extra nodes for them
                StreamAccessKind::SlidingWindow(_) => None,
                _ => Some((
                    Node::Stream(source_ref),
                    EdgeType::from(*kind),
                    Node::Stream(*target_ref),
                )),
            })
        })
    });

    let window_edges = mir.sliding_windows.iter().flat_map(|window| {
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

    access_edges.chain(window_edges).collect()
}

impl<'a> dot::Labeller<'a, Node, Edge> for DependencyGraph<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("example1").unwrap()
    }

    fn node_id(&'a self, n: &Node) -> dot::Id<'a> {
        let id = match n {
            Node::Stream(StreamReference::In(i)) => format!("In_{i}"),
            Node::Stream(StreamReference::Out(i)) => format!("Out_{i}"),
            Node::Window(WindowReference::Sliding(i)) => format!("W_{i}"),
            Node::Window(_) => unimplemented!(),
            Node::Trigger(i) => format!("T_{i}")
        };

        dot::Id::new(id).unwrap()
    }

    fn node_label<'b>(&'b self, n: &Node) -> LabelText<'b> {
        let infos = self.infos.get(n).unwrap();

        let label_text = match infos {
            NodeInformation::Stream {
                stream_name,
                eval_layer,
                memory_bound,
                pacing_ty,
                value_ty,
                kind: _,
            } => {
                format!(
                    "{stream_name}: {value_ty}\nPacing: {pacing_ty}\nMemory Bound: {memory_bound}\nLayer {eval_layer}"
                )
            },
            NodeInformation::Window => todo!(),
            NodeInformation::Trigger {
                idx,
                eval_layer,
                pacing_ty,
                value_ty,
                message,
            } => format!("Trigger {idx}: {value_ty}\nPacing: {pacing_ty}\nLayer: {eval_layer}\n\n{message}")
        };

        dot::LabelText::HtmlStr(label_text.into())
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
        };
        dot::LabelText::LabelStr(label.into())
    }

    fn edge_style<'b>(&'b self, edge: &Edge) -> Style {
        let kind = &edge.1;
        match kind {
            EdgeType::Access(StreamAccessKind::Hold) => Style::Dashed,
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
            Node::Trigger(_) => "ellipse",
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

impl<'a> dot::GraphWalk<'a, Node, Edge> for DependencyGraph<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Node> {
        Cow::Borrowed(&self.nodes)
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge> {
        Cow::Borrowed(&self.edges)
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
        let dep_graph = mir.dependency_graph();

        println!("{}", dep_graph.dot());
    }
}
