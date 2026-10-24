//! Render settings panel, laid out like MagicaVoxel's render tabs
//! (Light, Camera/Film, Sample, Display).

use std::ops::RangeInclusive;
use std::path::PathBuf;

use egui::{CollapsingHeader, ComboBox, Slider, Ui};
use glam::Vec3;

use crate::edit::{Brush, Editor, Tool};
use crate::env::EnvMode;
use crate::scene::{GpuMaterial, MaterialKind, Projection, RenderSettings, Scene};

/// Things the panel asks the app to do this frame.
#[derive(Default)]
pub struct Actions {
    pub reset_camera: bool,
    pub snapshot: bool,
    /// Area left between the panels for the 3D view, in points.
    pub viewport: Option<egui::Rect>,
}

pub struct PanelState {
    pub ibl_files: Vec<PathBuf>,
}

fn slider(ui: &mut Ui, label: &str, v: &mut f32, range: RangeInclusive<f32>) {
    ui.add(Slider::new(v, range).text(label));
}

fn slider_u(ui: &mut Ui, label: &str, v: &mut u32, range: RangeInclusive<u32>) {
    ui.add(Slider::new(v, range).text(label));
}

fn color(ui: &mut Ui, label: &str, c: &mut Vec3) {
    ui.horizontal(|ui| {
        let mut rgb = c.to_array();
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            *c = Vec3::from_array(rgb);
        }
        ui.label(label);
    });
}

pub fn settings_panel(ui: &mut Ui, s: &mut RenderSettings, state: &PanelState, spp: u32, size: (u32, u32)) -> Actions {
    let mut actions = Actions::default();
    ui.horizontal(|ui| {
        ui.label(format!("{spp} spp · {}×{}", size.0, size.1));
    });
    ui.horizontal(|ui| {
        actions.reset_camera = ui.button("Reset camera").clicked();
        actions.snapshot = ui.button("Snapshot").clicked();
    });
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        CollapsingHeader::new("Sun").default_open(true).show(ui, |ui| {
            slider(ui, "Pitch", &mut s.sun_pitch, -90.0..=90.0);
            slider(ui, "Yaw", &mut s.sun_yaw, 0.0..=360.0);
            slider(ui, "Area", &mut s.sun_area, 0.0..=1.0);
            slider(ui, "Intensity", &mut s.sun_intensity, 0.0..=5.0);
            color(ui, "Color", &mut s.sun_color);
            ui.checkbox(&mut s.sun_disk, "Show sun disk");
        });

        CollapsingHeader::new("Sky").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut s.env_mode, EnvMode::Uniform, "Uniform");
                ui.selectable_value(&mut s.env_mode, EnvMode::Atmosphere, "Atmosphere");
                ui.selectable_value(&mut s.env_mode, EnvMode::Image, "Image");
            });
            match s.env_mode {
                EnvMode::Uniform => {
                    slider(ui, "Intensity", &mut s.sky_intensity, 0.0..=5.0);
                    color(ui, "Color", &mut s.sky_color);
                }
                EnvMode::Atmosphere => {
                    slider(ui, "Rayleigh", &mut s.rayleigh_density, 0.0..=2.0);
                    color(ui, "Rayleigh color", &mut s.rayleigh_color);
                    slider(ui, "Mie", &mut s.mie_density, 0.0..=2.0);
                    slider(ui, "Mie phase", &mut s.mie_g, 0.5..=0.99);
                    color(ui, "Mie color", &mut s.mie_color);
                    slider(ui, "Ozone", &mut s.ozone_density, 0.0..=2.0);
                    color(ui, "Ozone color", &mut s.ozone_color);
                }
                EnvMode::Image => {
                    let name = |p: &PathBuf| p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let current = s.ibl_path.as_ref().map(name).unwrap_or_else(|| "none".into());
                    ComboBox::from_label("Image").selected_text(current).show_ui(ui, |ui| {
                        for f in &state.ibl_files {
                            ui.selectable_value(&mut s.ibl_path, Some(f.clone()), name(f));
                        }
                    });
                    slider(ui, "Intensity", &mut s.ibl_intensity, 0.0..=5.0);
                    slider(ui, "Rotation", &mut s.ibl_rotation, 0.0..=1.0);
                }
            }
        });

        CollapsingHeader::new("Fog").show(ui, |ui| {
            slider(ui, "Density", &mut s.fog_density, 0.0..=5.0);
            slider(ui, "Phase", &mut s.fog_g, -0.9..=0.9);
            color(ui, "Color", &mut s.fog_color);
        });

        CollapsingHeader::new("Lens").default_open(true).show(ui, |ui| {
            ComboBox::from_label("Projection").selected_text(format!("{:?}", s.projection)).show_ui(ui, |ui| {
                for p in [
                    Projection::Perspective,
                    Projection::Orthographic,
                    Projection::Stereographic,
                    Projection::Panorama,
                    Projection::Cylindrical,
                ] {
                    ui.selectable_value(&mut s.projection, p, format!("{p:?}"));
                }
            });
            slider(ui, "Field of view", &mut s.fov, 1.0..=170.0);
            slider(ui, "Aperture", &mut s.aperture, 0.0..=2.0);
            slider_u(ui, "Blades", &mut s.blades, 0..=8);
            slider(ui, "Blade rotation", &mut s.blade_rotation, -180.0..=180.0);
        });

        CollapsingHeader::new("Film").default_open(true).show(ui, |ui| {
            slider(ui, "Exposure", &mut s.exposure, 0.0..=8.0);
            slider(ui, "Vignette", &mut s.vignette, 0.0..=1.0);
            slider(ui, "Gamma", &mut s.gamma, 1.0..=5.0);
            ui.checkbox(&mut s.aces, "ACES tone mapping");
            ui.checkbox(&mut s.bloom, "Bloom");
            if s.bloom {
                slider(ui, "Mix", &mut s.bloom_mix, 0.0..=1.0);
                slider(ui, "Size", &mut s.bloom_scale, 0.0..=1.0);
                slider(ui, "Aspect", &mut s.bloom_aspect, 0.0..=1.0);
                slider(ui, "Threshold", &mut s.bloom_threshold, 0.0..=5.0);
            }
        });

        CollapsingHeader::new("Sample").show(ui, |ui| {
            slider_u(ui, "Diffuse depth", &mut s.max_diffuse, 0..=15);
            slider_u(ui, "Specular depth", &mut s.max_specular, 5..=15);
            slider_u(ui, "Scatter depth", &mut s.max_scatter, 5..=15);
            slider(ui, "Clamp energy", &mut s.clamp, 1.0..=5000.0);
            ui.checkbox(&mut s.translucent_shadows, "Translucent shadows");
        });

        CollapsingHeader::new("Display").show(ui, |ui| {
            ui.checkbox(&mut s.ground, "Ground");
            color(ui, "Ground color", &mut s.ground_color);
        });
    });
    actions
}

// ------------------------------------------------------------ editor panel


/// What the editor panel changed or asked for this frame.
#[derive(Default)]
pub struct EditActions {
    pub undo: bool,
    pub redo: bool,
    pub save: bool,
    /// Palette or material edits that need uploading.
    pub materials_changed: bool,
}

const KINDS: [(MaterialKind, &str); 6] = [
    (MaterialKind::Diffuse, "Diffuse"),
    (MaterialKind::Metal, "Metal"),
    (MaterialKind::Glass, "Glass"),
    (MaterialKind::Emit, "Emit"),
    (MaterialKind::Blend, "Blend"),
    (MaterialKind::Media, "Cloud"),
];

fn material_editor(ui: &mut Ui, m: &mut GpuMaterial) {
    let name = KINDS.iter().find(|(k, _)| *k as u32 == m.kind).map(|(_, n)| *n).unwrap_or("?");
    ComboBox::from_label("Material").selected_text(name).show_ui(ui, |ui| {
        for (k, n) in KINDS {
            ui.selectable_value(&mut m.kind, k as u32, n);
        }
    });
    let kind = m.kind;
    let is = |k: MaterialKind| kind == k as u32;
    if is(MaterialKind::Metal) || is(MaterialKind::Glass) || is(MaterialKind::Blend) {
        slider(ui, "Roughness", &mut m.rough, 0.0..=1.0);
        slider(ui, "IOR", &mut m.ior, 1.0..=3.0);
    }
    if is(MaterialKind::Metal) || is(MaterialKind::Blend) {
        slider(ui, "Metallic", &mut m.metal, 0.0..=1.0);
    }
    if is(MaterialKind::Glass) || is(MaterialKind::Blend) {
        slider(ui, "Transparency", &mut m.trans, 0.0..=1.0);
    }
    if is(MaterialKind::Emit) {
        ui.add(Slider::new(&mut m.emit, 0.0..=100.0).logarithmic(true).text("Emission"));
    }
    if is(MaterialKind::Media) {
        slider(ui, "Density", &mut m.density, 0.0..=10.0);
        slider(ui, "Phase", &mut m.g, -0.9..=0.9);
    }
}

/// Left panel. Brushes only show in the modeler and materials only in the
/// renderer, like MagicaVoxel's two modes.
pub fn editor_panel(ui: &mut Ui, e: &mut Editor, scene: &mut Scene, modeling: bool) -> EditActions {
    let mut a = EditActions::default();
    if modeling {
        brush_controls(ui, e);
    }
    ui.horizontal(|ui| {
        a.undo = ui.add_enabled(e.can_undo(), egui::Button::new("Undo")).clicked();
        a.redo = ui.add_enabled(e.can_redo(), egui::Button::new("Redo")).clicked();
        a.save = ui.button("Save").on_hover_text("Ctrl+S").clicked();
    });
    ui.separator();
    palette_controls(ui, e, scene, modeling, &mut a);
    a
}

fn brush_controls(ui: &mut Ui, e: &mut Editor) {
    ui.horizontal(|ui| {
        ui.selectable_value(&mut e.tool, Tool::Attach, "Attach").on_hover_text("T");
        ui.selectable_value(&mut e.tool, Tool::Erase, "Erase").on_hover_text("R");
        ui.selectable_value(&mut e.tool, Tool::Paint, "Paint").on_hover_text("G");
    });
    ui.horizontal(|ui| {
        ui.selectable_value(&mut e.brush, Brush::Voxel, "Voxel").on_hover_text("V");
        ui.selectable_value(&mut e.brush, Brush::Face, "Face").on_hover_text("F");
        ui.selectable_value(&mut e.brush, Brush::Box, "Box").on_hover_text("B");
        ui.selectable_value(&mut e.brush, Brush::Line, "Line").on_hover_text("L");
    });
    ui.horizontal(|ui| {
        ui.label("Mirror");
        for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
            ui.toggle_value(&mut e.mirror[i], *n).on_hover_text(format!("F{}", 5 + i));
        }
    });
    ui.separator();
}

fn palette_controls(ui: &mut Ui, e: &mut Editor, scene: &mut Scene, modeling: bool, a: &mut EditActions) {

    // Palette: 8 columns like MagicaVoxel, color index 1..=255.
    let swatch = 16.0;
    egui::Grid::new("palette").spacing([2.0, 2.0]).min_col_width(swatch).show(ui, |ui| {
        for i in 1..=255usize {
            let p = scene.palette[i];
            let fill = egui::Color32::from_rgb(p[0], p[1], p[2]);
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(swatch, swatch), egui::Sense::click());
            ui.painter().rect_filled(rect, 2.0, fill);
            if e.color as usize == i {
                ui.painter().rect_stroke(rect.expand(1.0), 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Outside);
            }
            if resp.clicked() {
                e.color = i as u8;
            }
            resp.on_hover_text(format!("#{i}"));
            if i % 8 == 0 {
                ui.end_row();
            }
        }
    });
    ui.separator();

    let i = e.color as usize;
    ui.horizontal(|ui| {
        let mut rgb = [scene.palette[i][0], scene.palette[i][1], scene.palette[i][2]];
        if ui.color_edit_button_srgb(&mut rgb).changed() {
            scene.palette[i] = [rgb[0], rgb[1], rgb[2], scene.palette[i][3]];
            scene.sync_color(i);
            a.materials_changed = true;
        }
        ui.label(format!("Color #{i}"));
    });
    if !modeling {
        let before = scene.materials[i];
        material_editor(ui, &mut scene.materials[i]);
        if bytemuck::bytes_of(&before) != bytemuck::bytes_of(&scene.materials[i]) {
            a.materials_changed = true;
        }
    }
}

/// Right panel in the modeler: MagicaVoxel's modeler "Display" options.
pub fn display_panel(ui: &mut Ui, o: &mut crate::modeler::ModelerOptions) -> Actions {
    let mut actions = Actions::default();
    ui.horizontal(|ui| {
        actions.reset_camera = ui.button("Reset camera").on_hover_text("F12").clicked();
    });
    ui.separator();
    CollapsingHeader::new("Display").default_open(true).show(ui, |ui| {
        ui.checkbox(&mut o.ao, "Ambient occlusion");
        ui.checkbox(&mut o.grid, "Grid").on_hover_text("Ctrl+G");
        ui.checkbox(&mut o.frame, "Frame").on_hover_text("Ctrl+F");
        ui.checkbox(&mut o.bounds, "Bounding box").on_hover_text("Ctrl+B");
        ui.checkbox(&mut o.constant, "Constant shading");
        let mut bg = o.background.to_array();
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut bg);
            ui.label("Background");
        });
        o.background = bg.into();
    });
    actions
}
