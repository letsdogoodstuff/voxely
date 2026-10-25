//! Importing models made elsewhere: other voxel formats, meshes,
//! heightmaps, slice sheets and images.

pub mod formats;
pub mod mesh;

use glam::IVec3;

/// Voxels read from another format, in voxely's axes (z up, a model
/// faces -y), with their own colors.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub name: String,
    pub dims: IVec3,
    /// Where this model's box sits among the file's other models.
    pub offset: IVec3,
    /// Positions inside 0..dims, colors RGBA (alpha 255).
    pub voxels: Vec<(IVec3, [u8; 4])>,
}
