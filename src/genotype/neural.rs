//! Neural control system representation.
//!
//! The brain is a dataflow graph of sensors, neurons, and effectors.
//! Each body part has its own local neural graph that gets instantiated
//! when the phenotype is built.

/// Sensor types that provide input signals
#[derive(Debug, Clone)]
pub enum SensorType {
    /// Current angle of a joint degree of freedom
    JointAngle { dof: usize },
    /// Contact sensor for a face of the part (activates on collision)
    Contact { face: Face },
    /// Photosensor - reacts to light source direction
    PhotoSensor { axis: SensorAxis },
}

/// Which face of a box-shaped part
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

/// Axis for photosensors (renamed to avoid bevy conflict)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorAxis {
    X,
    Y,
    Z,
}

/// Neuron function types from Karl Sims' paper
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronFunc {
    // Basic arithmetic
    Sum,
    Product,
    Divide,

    // Comparison
    SumThreshold,
    GreaterThan,
    SignOf,
    Min,
    Max,
    Abs,
    If,

    // Interpolation
    Interpolate,

    // Trigonometric
    Sin,
    Cos,
    Atan,

    // Mathematical
    Log,
    Exp,
    Sigmoid,

    // Temporal (stateful)
    Integrate,
    Differentiate,
    Smooth,
    Memory,

    // Oscillators (stateful)
    OscillateWave,
    OscillateSaw,
}

impl NeuronFunc {
    /// Number of inputs this function expects
    pub fn num_inputs(&self) -> usize {
        match self {
            // Unary functions
            NeuronFunc::SignOf
            | NeuronFunc::Abs
            | NeuronFunc::Sin
            | NeuronFunc::Cos
            | NeuronFunc::Atan
            | NeuronFunc::Log
            | NeuronFunc::Exp
            | NeuronFunc::Sigmoid
            | NeuronFunc::Integrate
            | NeuronFunc::Differentiate
            | NeuronFunc::Smooth
            | NeuronFunc::Memory
            | NeuronFunc::OscillateWave
            | NeuronFunc::OscillateSaw => 1,

            // Binary functions
            NeuronFunc::Sum
            | NeuronFunc::Product
            | NeuronFunc::Divide
            | NeuronFunc::SumThreshold
            | NeuronFunc::GreaterThan
            | NeuronFunc::Min
            | NeuronFunc::Max => 2,

            // Ternary functions
            NeuronFunc::If | NeuronFunc::Interpolate => 3,
        }
    }
}

/// Input source for a neuron
#[derive(Debug, Clone)]
pub enum NeuralInput {
    /// Constant value
    Constant(f32),
    /// From a sensor in the same part
    Sensor(usize),
    /// From another neuron in the same part
    LocalNeuron(usize),
    /// From a neuron in the parent part
    ParentNeuron(usize),
    /// From a neuron in a child part (by connection index)
    ChildNeuron { connection: usize, neuron: usize },
    /// From a centralized (unassociated) neuron
    CentralNeuron(usize),
}

/// A weighted input connection
#[derive(Debug, Clone)]
pub struct WeightedInput {
    pub source: NeuralInput,
    pub weight: f32,
}

/// A neuron that processes inputs to produce an output
#[derive(Debug, Clone)]
pub struct Neuron {
    pub func: NeuronFunc,
    pub inputs: Vec<WeightedInput>,
}

/// An effector that applies force to a joint DOF
#[derive(Debug, Clone)]
pub struct Effector {
    /// Which degree of freedom of this part's joint
    pub dof: usize,
    /// Input source and weight
    pub input: WeightedInput,
    /// Maximum force this effector can apply
    pub max_force: f32,
}

/// Neural graph for a single body part
#[derive(Debug, Clone)]
pub struct NeuralGraph {
    pub sensors: Vec<SensorType>,
    pub neurons: Vec<Neuron>,
    pub effectors: Vec<Effector>,
}

impl NeuralGraph {
    pub fn new() -> Self {
        Self {
            sensors: Vec::new(),
            neurons: Vec::new(),
            effectors: Vec::new(),
        }
    }

    pub fn add_sensor(&mut self, sensor: SensorType) -> usize {
        let idx = self.sensors.len();
        self.sensors.push(sensor);
        idx
    }

    pub fn add_neuron(&mut self, neuron: Neuron) -> usize {
        let idx = self.neurons.len();
        self.neurons.push(neuron);
        idx
    }

    pub fn add_effector(&mut self, effector: Effector) {
        self.effectors.push(effector);
    }
}

impl Default for NeuralGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Centralized neurons not associated with any body part
#[derive(Debug, Clone)]
pub struct CentralNervousSystem {
    pub neurons: Vec<Neuron>,
}

impl CentralNervousSystem {
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
        }
    }
}

impl Default for CentralNervousSystem {
    fn default() -> Self {
        Self::new()
    }
}
