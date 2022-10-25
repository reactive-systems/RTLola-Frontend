use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Display;
use std::io::BufWriter;

use dot::{LabelText, Style};
use serde::{Serialize, Serializer};
use serde_json::{json, to_string_pretty};

use super::{Mir, StreamAccessKind, StreamReference, TriggerReference, WindowReference};

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
        let trigger_nodes = mir
            .triggers
            .iter()
            .map(|trigger| Node::Trigger(trigger.trigger_reference));
        let nodes: Vec<_> = stream_nodes.chain(window_nodes).chain(trigger_nodes).collect();

        let edges = edges(mir);

        let infos = nodes.iter().map(|node| (*node, node_infos(mir, *node))).collect();

        Self { nodes, edges, infos }
    }

    /// Returns the dependency graph in the graphviz dot-format
    pub fn dot(&self) -> String {
        let res = Vec::new();
        let mut res_writer = BufWriter::new(res);
        dot::render(self, &mut res_writer).unwrap();
        String::from_utf8(res_writer.into_inner().unwrap()).unwrap()
    }

    /// Returns the dependency graph in a json-format
    pub fn json(&self) -> String {
        let infos = self
            .infos
            .iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect::<HashMap<_, _>>();

        let json_value = json!({
            "edges": self.edges,
            "nodes": infos
        });

        to_string_pretty(&json_value).unwrap()
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

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Stream(StreamReference::In(i)) => write!(f, "In_{i}"),
            Node::Stream(StreamReference::Out(i)) => write!(f, "Out_{i}"),
            Node::Window(WindowReference::Sliding(i)) => write!(f, "W_{i}"),
            Node::Window(_) => unimplemented!(),
            Node::Trigger(i) => write!(f, "T_{i}"),
        }
    }
}

impl Serialize for Node {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

#[derive(Clone, Debug, Serialize)]
struct Edge {
    from: Node,
    with: EdgeType,
    to: Node,
}

#[derive(Clone, Debug)]
enum EdgeType {
    Access(StreamAccessKind),
}

impl From<StreamAccessKind> for EdgeType {
    fn from(ak: StreamAccessKind) -> Self {
        EdgeType::Access(ak)
    }
}

impl Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EdgeType::Access(sak) => {
                match sak {
                    StreamAccessKind::Sync => "Sync".into(),
                    StreamAccessKind::Hold => "Hold".into(),
                    StreamAccessKind::Offset(o) => format!("Offset({o})"),
                    StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => "".into(),
                    StreamAccessKind::Get => todo!(),
                    StreamAccessKind::Fresh => todo!(),
                }
            },
        };

        write!(f, "{s}")
    }
}

impl Serialize for EdgeType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

#[derive(Serialize, Debug, Clone)]
#[serde(untagged)]
enum NodeInformation<'a> {
    Stream {
        kind: StreamKind,
        stream_name: &'a str,
        eval_layer: usize,
        memory_bound: u32,
        pacing_ty: String,
        value_ty: String,
        expression: String,
    },

    Trigger {
        idx: usize,
        eval_layer: usize,
        pacing_ty: String,
        value_ty: String,
        message: &'a str,
        expression: String,
    },

    Window {
        idx: usize,
        operation: String,
        duration: String,
    },
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
        mir.display(pacing_ty).to_string()
    } else {
        "input".into()
    };

    let expression_str = if let StreamReference::Out(_) = sref {
        mir.display(&mir.output(sref).eval.expression).to_string()
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
        expression: expression_str,
    }
}

fn window_infos(mir: &Mir, wref: WindowReference) -> NodeInformation {
    let idx = wref.idx();
    let window = mir.sliding_window(wref);
    let operation_str = window.op.to_string();
    let duration_str = format!("{}s", window.duration.as_secs_f32());

    NodeInformation::Window {
        idx,
        operation: operation_str,
        duration: duration_str,
    }
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
        expression,
    } = stream_infos(mir, trigger.reference)
    {
        NodeInformation::Trigger {
            idx: tref,
            eval_layer,
            pacing_ty,
            value_ty,
            message: &trigger.message,
            expression,
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
            access_kinds.iter().filter_map(move |kind| {
                match kind {
                    // we remove edges for window accesses, because we add extra nodes for them
                    StreamAccessKind::SlidingWindow(_) => None,
                    _ => {
                        Some(Edge {
                            from: Node::Stream(source_ref),
                            with: EdgeType::from(*kind),
                            to: Node::Stream(*target_ref),
                        })
                    },
                }
            })
        })
    });

    let window_edges = mir.sliding_windows.iter().flat_map(|window| {
        [
            Edge {
                from: Node::Window(window.reference),
                with: StreamAccessKind::SlidingWindow(window.reference).into(),
                to: Node::Stream(window.caller),
            },
            Edge {
                from: Node::Stream(window.target),
                with: StreamAccessKind::SlidingWindow(window.reference).into(),
                to: Node::Window(window.reference),
            },
        ]
    });

    let trigger_edges = mir.triggers.iter().map(|trigger| {
        Edge {
            from: Node::Stream(trigger.reference),
            with: StreamAccessKind::Sync.into(),
            to: Node::Trigger(trigger.trigger_reference),
        }
    });

    access_edges.chain(window_edges).chain(trigger_edges).collect()
}

impl<'a> dot::Labeller<'a, Node, Edge> for DependencyGraph<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("example1").unwrap()
    }

    fn node_id(&'a self, n: &Node) -> dot::Id<'a> {
        let id = n.to_string();
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
                expression: _,
            } => {
                format!(
                    "{stream_name}: {value_ty}<br/>Pacing: {pacing_ty}<br/>Memory Bound: {memory_bound}<br/>Layer {eval_layer}"
                )
            },
            NodeInformation::Window {
                idx,
                operation,
                duration,
            } => format!("Window {idx}<br/>Window Operation: {operation}<br/>Duration: {duration}"),
            NodeInformation::Trigger {
                idx,
                eval_layer,
                pacing_ty,
                value_ty,
                message,
                expression: _,
            } => {
                format!("Trigger {idx}: {value_ty}<br/>Pacing: {pacing_ty}<br/>Layer: {eval_layer}<br/><br/>{message}")
            },
        };

        dot::LabelText::HtmlStr(label_text.into())
    }

    fn edge_label<'b>(&'b self, edge: &Edge) -> LabelText<'b> {
        let kind = &edge.with;
        let label = match kind {
            EdgeType::Access(sak) => {
                match sak {
                    StreamAccessKind::Sync => "Sync".into(),
                    StreamAccessKind::Hold => "Hold".into(),
                    StreamAccessKind::Offset(o) => format!("Offset({o})"),
                    StreamAccessKind::DiscreteWindow(_) | StreamAccessKind::SlidingWindow(_) => "".into(),
                    StreamAccessKind::Get => todo!(),
                    StreamAccessKind::Fresh => todo!(),
                }
            },
        };
        dot::LabelText::LabelStr(label.into())
    }

    fn edge_style<'b>(&'b self, edge: &Edge) -> Style {
        let kind = &edge.with;
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
            Node::Trigger(_) => "octagon",
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
        e.from
    }

    fn target(&self, e: &Edge) -> Node {
        e.to
    }
}
