//! MagicaVoxel's voxel shader ("Code") brush.
//!
//! A shader is a GLSL snippet defining `float map(vec3 v)` that returns the
//! new color index for the voxel centered at `v`. A header between
//! `// xs_begin` and `// xs_end` declares float arguments:
//!
//! ```text
//! // arg : { var = 'm_radius' name = 'Radius' value = '3' range = '1 128' step = '1' precision = '0' }
//! ```
//!
//! The snippet is wrapped in a GLSL compute shader, compiled with naga's
//! GLSL front end and run on the GPU over the whole volume. Built-ins:
//! `i_volume_size`, `i_color_index`, `i_num_color_sels`, `i_args[8]`,
//! `voxel(v)`, `color_sel(k)` and `palette(index)`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use wgpu::util::DeviceExt;

use crate::scene::Volume;

#[derive(Clone, Debug, PartialEq)]
pub struct XsArg {
    /// Uniform name used in the snippet, e.g. `m_radius`.
    pub var: String,
    /// Label shown in the UI.
    pub name: String,
    /// Default value.
    pub value: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    /// Decimal places to display.
    pub precision: u32,
}

#[derive(Clone, Debug)]
pub struct XsShader {
    /// File stem, e.g. `torus`.
    pub name: String,
    pub source: String,
    pub args: Vec<XsArg>,
}

impl XsShader {
    pub fn load(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        Self::parse(&name, &source)
    }

    pub fn parse(name: &str, source: &str) -> Result<Self> {
        let mut args = Vec::new();
        let mut in_header = false;
        for line in source.lines() {
            let t = line.trim();
            let Some(c) = t.strip_prefix("//") else { continue };
            let c = c.trim();
            if c.starts_with("xs_begin") {
                in_header = true;
            } else if c.starts_with("xs_end") {
                break;
            } else if in_header {
                if let Some(rest) = c.strip_prefix("arg") {
                    let rest = rest.trim_start().strip_prefix(':').unwrap_or(rest);
                    let arg = parse_arg(rest).with_context(|| format!("{name}: bad arg line `{t}`"))?;
                    if args.iter().any(|a: &XsArg| a.var == arg.var) {
                        bail!("{name}: duplicate arg `{}`", arg.var);
                    }
                    args.push(arg);
                }
            }
        }
        if !source.contains("map") {
            bail!("{name}: no `float map(vec3 v)` function");
        }
        Ok(XsShader { name: name.to_string(), source: source.to_string(), args })
    }
}

/// Parse `{ key = 'value' key = 'value' ... }`.
fn parse_arg(s: &str) -> Result<XsArg> {
    let mut fields = Vec::new();
    let mut rest = s.trim().trim_start_matches('{');
    loop {
        rest = rest.trim_start();
        if rest.is_empty() || rest.starts_with('}') {
            break;
        }
        let eq = rest.find('=').context("expected `key = 'value'`")?;
        let key = rest[..eq].trim().to_string();
        let after = rest[eq + 1..].trim_start();
        let quote = after.chars().next().context("missing value")?;
        if quote != '\'' && quote != '"' {
            bail!("value of `{key}` must be quoted");
        }
        let end = after[1..].find(quote).context("unterminated value")? + 1;
        fields.push((key, after[1..end].to_string()));
        rest = &after[end + 1..];
    }
    let get = |k: &str| fields.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
    let num = |k: &str, d: f32| -> Result<f32> {
        match get(k) {
            Some(v) => v.trim().parse().with_context(|| format!("`{k}` is not a number")),
            None => Ok(d),
        }
    };
    let var = get("var").context("missing `var`")?.trim().to_string();
    if var.is_empty() || !var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || var.starts_with(|c: char| c.is_ascii_digit()) {
        bail!("`{var}` is not a valid identifier");
    }
    let value = num("value", 0.0)?;
    let (min, max) = match get("range") {
        Some(r) => {
            let v: Vec<f32> = r.split_whitespace().map(str::parse).collect::<Result<_, _>>().context("bad `range`")?;
            match v[..] {
                [a, b] => (a.min(b), a.max(b)),
                _ => bail!("`range` needs two numbers"),
            }
        }
        None => (f32::MIN, f32::MAX),
    };
    Ok(XsArg {
        name: get("name").map(str::to_string).unwrap_or_else(|| var.clone()),
        var,
        value,
        min,
        max,
        step: num("step", 1.0)?,
        precision: num("precision", 0.0)?.max(0.0) as u32,
    })
}

/// Sorted `*.txt` files in `dir` (empty if it can't be read).
#[allow(dead_code)] // used by the UI
pub fn list_shaders(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt")))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

const WG: u32 = 4;
/// Palette entries, then up to 256 selected colors, in one buffer.
const TABLE_LEN: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Header {
    /// xyz volume size, w number of color selections.
    dims: [i32; 4],
    /// z = first slice of this slab, w = slab depth.
    origin: [i32; 4],
    /// x = current color index.
    misc: [f32; 4],
    i_args: [f32; 8],
}

/// GLSL wrapped around the user's snippet; `{ARGS}` and `{INIT}` are filled in.
const PRELUDE: &str = r#"#version 450
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std140, set = 0, binding = 0) uniform XsParams {
    ivec4 dims;
    ivec4 origin;
    vec4 misc;
    vec4 i_args_v[2];
    vec4 m_args_v[{NV}];
} xs_p;
layout(std430, set = 0, binding = 1) readonly buffer XsSrc { uint data[]; } xs_src;
layout(std430, set = 0, binding = 2) buffer XsDst { uint data[]; } xs_dst;
layout(std430, set = 0, binding = 3) readonly buffer XsTable { uint data[]; } xs_table;

vec3 i_volume_size;
float i_color_index;
int i_num_color_sels;
float i_args[8];
{ARGS}

float voxel(vec3 v) {
    ivec3 p = ivec3(floor(v));
    if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, xs_p.dims.xyz))) {
        return 0.0;
    }
    uint i = uint(p.x) + uint(xs_p.dims.x) * (uint(p.y) + uint(xs_p.dims.y) * uint(p.z));
    return float((xs_src.data[i >> 2u] >> ((i & 3u) * 8u)) & 255u);
}

float color_sel(float k) {
    int i = clamp(int(floor(k)), 0, max(i_num_color_sels, 1) - 1);
    return float(xs_table.data[256 + i]);
}

vec4 palette(float index) {
    uint c = xs_table.data[clamp(int(floor(index + 0.5)), 0, 255)];
    return vec4(float(c & 255u), float((c >> 8u) & 255u), float((c >> 16u) & 255u), float(c >> 24u)) / 255.0;
}

// ---- user snippet ----
"#;

const MAIN: &str = r#"
// ---- end of user snippet ----

void main() {
    ivec3 g = ivec3(gl_GlobalInvocationID);
    ivec3 p = g + ivec3(0, 0, xs_p.origin.z);
    if (g.z >= xs_p.origin.w || any(greaterThanEqual(p, xs_p.dims.xyz))) {
        return;
    }
    i_volume_size = vec3(xs_p.dims.xyz);
    i_color_index = xs_p.misc.x;
    i_num_color_sels = xs_p.dims.w;
    for (int k = 0; k < 8; k++) {
        i_args[k] = xs_p.i_args_v[k / 4][k % 4];
    }
{INIT}
    float r = map(vec3(p) + vec3(0.5));
    uint res = uint(clamp(floor(r + 0.5), 0.0, 255.0));
    uint i = uint(g.x) + uint(xs_p.dims.x) * (uint(g.y) + uint(xs_p.dims.y) * uint(g.z));
    xs_dst.data[i] = res;
}
"#;

fn wrap(shader: &XsShader, snippet: &str) -> String {
    let nv = shader.args.len().div_ceil(4).max(1);
    let decls: String = shader.args.iter().map(|a| format!("float {};\n", a.var)).collect();
    let init: String = shader
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| format!("    {} = xs_p.m_args_v[{}][{}];\n", a.var, i / 4, i % 4))
        .collect();
    let mut s = PRELUDE.replace("{NV}", &nv.to_string()).replace("{ARGS}", &decls);
    let user_line = s.lines().count() + 1;
    s.push_str(&format!("#line {user_line}\n"));
    s.push_str(snippet);
    s.push_str(&MAIN.replace("{INIT}", &init));
    s
}

/// Lenient fixups for code MagicaVoxel's GL driver accepted: a `float`
/// initialized from an expression containing a vector constructor (e.g.
/// gear.txt's `float col = d * vec3( 1.-step(f,R)) ;`) takes the
/// constructor's first component, which for a splat is the same value.
fn fixup(src: &str) -> Option<String> {
    let mut changed = false;
    let out: Vec<String> = src
        .lines()
        .map(|line| {
            let t = line.trim_start();
            let is_float_decl = t.strip_prefix("float").is_some_and(|r| r.starts_with(char::is_whitespace) && r.contains('='));
            if is_float_decl && ["vec2(", "vec3(", "vec4(", "vec2 (", "vec3 (", "vec4 ("].iter().any(|v| t.contains(v)) {
                changed = true;
                let mut l = line.to_string();
                for n in ["2", "3", "4"] {
                    l = l.replace(&format!("vec{n}("), "float(").replace(&format!("vec{n} ("), "float(");
                }
                l
            } else {
                line.to_string()
            }
        })
        .collect();
    changed.then(|| out.join("\n"))
}

fn compile_module(name: &str, glsl: &str) -> Result<naga::Module, String> {
    let mut fe = naga::front::glsl::Frontend::default();
    let opts = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
    let module = fe.parse(&opts, glsl).map_err(|e| e.emit_to_string_with_path(glsl, name))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .map_err(|e| e.emit_to_string_with_path(glsl, name))?;
    Ok(module)
}

/// GLSL for `shader`, trying the lenient fixups if the source as written
/// doesn't compile.
fn translate(shader: &XsShader) -> Result<(String, naga::Module)> {
    let glsl = wrap(shader, &shader.source);
    let err = match compile_module(&shader.name, &glsl) {
        Ok(m) => return Ok((glsl, m)),
        Err(e) => e,
    };
    if let Some(fixed) = fixup(&shader.source) {
        let glsl = wrap(shader, &fixed);
        if let Ok(m) = compile_module(&shader.name, &glsl) {
            log::warn!("xs shader {}: compiled after lenient fixups", shader.name);
            return Ok((glsl, m));
        }
    }
    bail!("voxel shader `{}` failed to compile:\n{err}", shader.name)
}

struct Compiled {
    device: wgpu::Device,
    key: String,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

static CACHE: Mutex<Vec<Compiled>> = Mutex::new(Vec::new());
const CACHE_MAX: usize = 16;

fn pipeline(device: &wgpu::Device, shader: &XsShader) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    let key = wrap(shader, &shader.source);
    {
        let cache = CACHE.lock().unwrap();
        if let Some(c) = cache.iter().find(|c| &c.device == device && c.key == key) {
            return Ok((c.pipeline.clone(), c.layout.clone()));
        }
    }
    let (_, module) = translate(shader)?;
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xs"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            storage(1, true),
            storage(2, false),
            storage(3, true),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("xs"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&shader.name), source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xs"),
        layout: Some(&pl),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        bail!("voxel shader `{}` failed to build: {e}", shader.name);
    }
    let mut cache = CACHE.lock().unwrap();
    if cache.len() >= CACHE_MAX {
        cache.remove(0);
    }
    cache.push(Compiled { device: device.clone(), key, pipeline: pipeline.clone(), layout: layout.clone() });
    Ok((pipeline, layout))
}

/// Run `shader` over the whole volume; `args` overrides arg values by index;
/// `color` is the current color index; `sels` the selected palette indices
/// (`[color]` if empty); `palette` 256 RGBA8 entries. Returns the new volume
/// (same dims) or a compile/run error naming the shader.
#[allow(clippy::too_many_arguments)]
pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader: &XsShader,
    args: &[f32],
    volume: &Volume,
    palette: &[[u8; 4]],
    color: u8,
    sels: &[u8],
) -> Result<Volume> {
    let dims = volume.dims;
    let n = volume.cells.len();
    if n == 0 {
        return Ok(volume.clone());
    }
    let limits = device.limits();
    let max_bytes = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
    let src_words = n.div_ceil(4);
    if (src_words as u64) * 4 > max_bytes {
        bail!("volume {}x{}x{} is too large for this GPU's storage buffers", dims.x, dims.y, dims.z);
    }
    let max_wg = limits.max_compute_workgroups_per_dimension;
    if (dims.x as u32).div_ceil(WG) > max_wg || (dims.y as u32).div_ceil(WG) > max_wg {
        bail!("volume {}x{}x{} exceeds the GPU's dispatch limits", dims.x, dims.y, dims.z);
    }
    let slice_cells = (dims.x * dims.y) as u64;
    // Output is u32 per cell, written in z slabs that fit the binding limit.
    let slab = ((max_bytes / 4 / slice_cells) as i32).min(dims.z).min((max_wg * WG) as i32);
    if slab < 1 {
        bail!("volume slice {}x{} is too large for this GPU", dims.x, dims.y);
    }

    let (pipeline, layout) = pipeline(device, shader)?;

    let mut packed = vec![0u32; src_words];
    for (i, &c) in volume.cells.iter().enumerate() {
        packed[i >> 2] |= (c as u32) << ((i & 3) * 8);
    }
    let sels: Vec<u8> = if sels.is_empty() { vec![color] } else { sels.iter().copied().take(256).collect() };
    let mut table = vec![0u32; TABLE_LEN];
    for (i, p) in palette.iter().take(256).enumerate() {
        table[i] = u32::from_le_bytes(*p);
    }
    for (i, &s) in sels.iter().enumerate() {
        table[256 + i] = s as u32;
    }
    let values: Vec<f32> = shader.args.iter().enumerate().map(|(i, a)| args.get(i).copied().unwrap_or(a.value)).collect();
    let mut i_args = [0f32; 8];
    for (d, v) in i_args.iter_mut().zip(&values) {
        *d = *v;
    }
    let mut m_args = vec![0f32; shader.args.len().div_ceil(4).max(1) * 4];
    m_args[..values.len()].copy_from_slice(&values);

    let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("xs_src"), contents: bytemuck::cast_slice(&packed), usage: wgpu::BufferUsages::STORAGE });
    let table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("xs_table"), contents: bytemuck::cast_slice(&table), usage: wgpu::BufferUsages::STORAGE });
    let slab_bytes = slice_cells * slab as u64 * 4;
    let dst_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("xs_dst"), size: slab_bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("xs_read"), size: slab_bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let mut cells = vec![0u8; n];
    let mut z0 = 0;
    while z0 < dims.z {
        let depth = slab.min(dims.z - z0);
        let header = Header { dims: [dims.x, dims.y, dims.z, sels.len() as i32], origin: [0, 0, z0, depth], misc: [color as f32, 0.0, 0.0, 0.0], i_args };
        let mut uniform = bytemuck::bytes_of(&header).to_vec();
        uniform.extend_from_slice(bytemuck::cast_slice(&m_args));
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("xs_params"), contents: &uniform, usage: wgpu::BufferUsages::UNIFORM });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xs"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dst_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: table_buf.as_entire_binding() },
            ],
        });
        let bytes = slice_cells * depth as u64 * 4;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("xs") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("xs"), timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((dims.x as u32).div_ceil(WG), (dims.y as u32).div_ceil(WG), (depth as u32).div_ceil(WG));
        }
        enc.copy_buffer_to_buffer(&dst_buf, 0, &read_buf, 0, bytes);
        queue.submit([enc.finish()]);
        let slice = read_buf.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| drop(tx.send(r)));
        device.poll(wgpu::PollType::wait_indefinitely()).with_context(|| format!("running voxel shader `{}`", shader.name))?;
        rx.recv().context("map callback dropped")?.with_context(|| format!("reading back voxel shader `{}`", shader.name))?;
        {
            let data = slice.get_mapped_range()?;
            let words: &[u32] = bytemuck::cast_slice(&data);
            let start = (slice_cells * z0 as u64) as usize;
            for (d, &w) in cells[start..start + words.len()].iter_mut().zip(words) {
                *d = w.min(255) as u8;
            }
        }
        read_buf.unmap();
        z0 += depth;
    }
    Ok(Volume::from_cells(dims, cells))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    const DIR: &str = "MagicaVoxel-0.99.6.2-macos-10.15/shader";

    fn shader_dir() -> Option<PathBuf> {
        let d = PathBuf::from(std::env::var("HOME").ok()?).join("Downloads").join(DIR);
        d.is_dir().then_some(d)
    }

    fn gpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let (_a, d, q) = pollster::block_on(crate::gpu::request_device(&instance, None)).expect("gpu");
        (d, q)
    }

    fn pal() -> Vec<[u8; 4]> {
        (0..256).map(|i| [i as u8, 0, 255 - i as u8, 255]).collect()
    }

    const TORUS_HEADER: &str = "// xs_begin\n// arg : { var = 'm_radius' name = 'Radius' value = '3'   range = '1 128'    step = '1'    precision = '0' }\n// xs_end\nfloat map(vec3 v) { return 0.0; }\n";
    const GEAR_HEADER: &str = "// xs_begin\n// author : '@mode_vis'\n// arg : { var = 'm_teeth' name = 'Teeth' value = '6'   range = '3 64'    step = '1'\tprecision = '0' }\n// arg : { var = 'm_inner' name = 'Inner' value = '0.5' range = '0.0 1.0' step = '0.05'\tprecision = '2' }\n// xs_end\nfloat map(vec3 v) { return 0.0; }\n";

    #[test]
    fn parses_args() {
        let t = XsShader::parse("torus", TORUS_HEADER).unwrap();
        assert_eq!(t.args, vec![XsArg { var: "m_radius".into(), name: "Radius".into(), value: 3.0, min: 1.0, max: 128.0, step: 1.0, precision: 0 }]);
        let g = XsShader::parse("gear", GEAR_HEADER).unwrap();
        assert_eq!(g.args.len(), 2);
        assert_eq!(g.args[0].var, "m_teeth");
        assert_eq!((g.args[0].value, g.args[0].min, g.args[0].max), (6.0, 3.0, 64.0));
        assert_eq!(g.args[1].name, "Inner");
        assert_eq!((g.args[1].value, g.args[1].min, g.args[1].max, g.args[1].step, g.args[1].precision), (0.5, 0.0, 1.0, 0.05, 2));
        // Args outside the header are ignored.
        let s = XsShader::parse("x", "// arg : { var = 'a' }\nfloat map(vec3 v){return 0.0;}").unwrap();
        assert!(s.args.is_empty());
    }

    #[test]
    fn fixup_vec_to_float() {
        let f = fixup("\tfloat col = d * vec3( 1.-step(f,R)) ;\n\tvec3 p = vec3(1.0);").unwrap();
        assert_eq!(f, "\tfloat col = d * float( 1.-step(f,R)) ;\n\tvec3 p = vec3(1.0);");
        assert!(fixup("vec3 p = vec3(1.0);").is_none());
    }

    #[test]
    fn bad_shader_reports_error() {
        let s = XsShader::parse("broken", "float map(vec3 v) { return undefined_thing; }").unwrap();
        let e = translate(&s).unwrap_err().to_string();
        assert!(e.contains("broken"), "{e}");
    }

    #[test]
    fn bundled_shaders_compile() {
        let Some(dir) = shader_dir() else { return };
        let files = list_shaders(&dir);
        assert_eq!(files.len(), 7);
        for f in files {
            let s = XsShader::load(&f).unwrap();
            translate(&s).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    #[test]
    fn gpu_sphere() {
        let (d, q) = gpu();
        let src = "float map( vec3 v ) {\n\tconst float PI = 3.1415926;\n\tv = ( v - i_volume_size * 0.5 ) / ( i_volume_size * 0.5 );\n\treturn i_color_index * step( length( v ), 1.0 );\n}\n";
        let s = XsShader::parse("sphere", src).unwrap();
        let vol = Volume::new(IVec3::splat(16));
        let out = run(&d, &q, &s, &[], &vol, &pal(), 7, &[]).unwrap();
        let count = out.cells.iter().filter(|&&c| c != 0).count();
        assert!(out.cells.iter().all(|&c| c == 0 || c == 7));
        // 4/3 pi 8^3 = 2145.
        assert!((1950..=2350).contains(&count), "{count}");
        // Center filled, corner empty.
        assert_eq!(out.get(IVec3::splat(8)), 7);
        assert_eq!(out.get(IVec3::ZERO), 0);
    }

    #[test]
    fn gpu_reads_voxels_and_builtins() {
        let (d, q) = gpu();
        let dims = IVec3::new(5, 7, 9);
        let mut vol = Volume::new(dims);
        for i in 0..vol.cells.len() {
            vol.cells[i] = (i % 5) as u8;
        }
        // Shift +x by one cell, add 10 to filled cells; x == 0 reads outside.
        let src = "// xs_begin\n// arg : { var = 'm_add' name = 'Add' value = '10' range = '0 100' }\n// xs_end\nfloat map(vec3 v) {\n  float c = voxel(v - vec3(1.0, 0.0, 0.0));\n  if (v.y > 6.0) return color_sel(v.x - 0.5);\n  if (v.z > 8.0) return palette(c).x * 255.0;\n  return c == 0.0 ? 0.0 : c + m_add + i_args[1];\n}\n";
        let s = XsShader::parse("shift", src).unwrap();
        let out = run(&d, &q, &s, &[20.0], &vol, &pal(), 3, &[9, 8, 7]).unwrap();
        for i in 0..vol.cells.len() {
            let p = vol.position(i);
            let c = vol.get(p - IVec3::X);
            let want = if p.y == 6 {
                [9, 8, 7][(p.x as usize).min(2)]
            } else if p.z == 8 {
                c
            } else if c == 0 {
                0
            } else {
                c + 20
            };
            assert_eq!(out.cells[i], want, "at {p}");
        }
    }

    #[test]
    fn gpu_bundled_shaders_run() {
        let Some(dir) = shader_dir() else { return };
        let (d, q) = gpu();
        let vol = Volume::new(IVec3::splat(32));
        for f in list_shaders(&dir) {
            let s = XsShader::load(&f).unwrap();
            let out = run(&d, &q, &s, &[], &vol, &pal(), 5, &[]).unwrap();
            let count = out.cells.iter().filter(|&&c| c != 0).count();
            assert!(count > 0, "{} produced nothing", s.name);
            // Cached second run gives the same result.
            let again = run(&d, &q, &s, &[], &vol, &pal(), 5, &[]).unwrap();
            assert_eq!(out.cells, again.cells);
        }
    }
}
