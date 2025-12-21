//! Generic directed graph using arena-style indexing.
//!
//! This avoids Rc<RefCell<>> complexity and is idiomatic Rust.

use std::ops::{Index, IndexMut};

/// Type-safe index into the node arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// A connection between two nodes
#[derive(Debug, Clone)]
pub struct Connection<C> {
    pub from: NodeId,
    pub to: NodeId,
    pub data: C,
}

/// A directed graph with nodes of type N and connection data of type C
#[derive(Debug, Clone)]
pub struct DirectedGraph<N, C> {
    nodes: Vec<N>,
    connections: Vec<Connection<C>>,
}

impl<N, C> DirectedGraph<N, C> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }

    /// Add a node and return its ID
    pub fn add_node(&mut self, node: N) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Add a connection between two nodes
    pub fn add_connection(&mut self, from: NodeId, to: NodeId, data: C) {
        self.connections.push(Connection { from, to, data });
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&N> {
        self.nodes.get(id.0)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut N> {
        self.nodes.get_mut(id.0)
    }

    /// Get all connections from a node
    pub fn connections_from(&self, id: NodeId) -> impl Iterator<Item = &Connection<C>> {
        self.connections.iter().filter(move |c| c.from == id)
    }

    /// Get all connections to a node
    pub fn connections_to(&self, id: NodeId) -> impl Iterator<Item = &Connection<C>> {
        self.connections.iter().filter(move |c| c.to == id)
    }

    /// Number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of connections
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Iterate over all nodes with their IDs
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &N)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (NodeId(i), n))
    }

    /// Iterate over all nodes mutably
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = (NodeId, &mut N)> {
        self.nodes
            .iter_mut()
            .enumerate()
            .map(|(i, n)| (NodeId(i), n))
    }

    /// Iterate over all connections
    pub fn connections(&self) -> impl Iterator<Item = &Connection<C>> {
        self.connections.iter()
    }

    /// Iterate over all connections mutably
    pub fn connections_mut(&mut self) -> impl Iterator<Item = &mut Connection<C>> {
        self.connections.iter_mut()
    }

    /// Check if a node ID is valid
    pub fn is_valid(&self, id: NodeId) -> bool {
        id.0 < self.nodes.len()
    }
}

impl<N, C> Default for DirectedGraph<N, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, C> Index<NodeId> for DirectedGraph<N, C> {
    type Output = N;

    fn index(&self, id: NodeId) -> &Self::Output {
        &self.nodes[id.0]
    }
}

impl<N, C> IndexMut<NodeId> for DirectedGraph<N, C> {
    fn index_mut(&mut self, id: NodeId) -> &mut Self::Output {
        &mut self.nodes[id.0]
    }
}
