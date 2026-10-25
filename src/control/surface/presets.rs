//! Well-known open palettes, with their published hex values.

pub struct Preset {
    pub name: &'static str,
    pub author: &'static str,
    pub colors: &'static [&'static str],
}

pub const PRESETS: &[Preset] = &[
    Preset {
        name: "pico-8",
        author: "Lexaloffle (PICO-8 fantasy console)",
        colors: &[
            "#000000", "#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f", "#c2c3c7", "#fff1e8", "#ff004d", "#ffa300", "#ffec27", "#00e436",
            "#29adff", "#83769c", "#ff77a8", "#ffccaa",
        ],
    },
    Preset {
        name: "sweetie-16",
        author: "GrafxKid (TIC-80's default palette)",
        colors: &[
            "#1a1c2c", "#5d275d", "#b13e53", "#ef7d57", "#ffcd75", "#a7f070", "#38b764", "#257179", "#29366f", "#3b5dc9", "#41a6f6", "#73eff7",
            "#f4f4f4", "#94b0c2", "#566c86", "#333c57",
        ],
    },
    Preset {
        name: "dawnbringer-16",
        author: "DawnBringer",
        colors: &[
            "#140c1c", "#442434", "#30346d", "#4e4a4e", "#854c30", "#346524", "#d04648", "#757161", "#597dce", "#d27d2c", "#8595a1", "#6daa2c",
            "#d2aa99", "#6dc2ca", "#dad45e", "#deeed6",
        ],
    },
    Preset {
        name: "dawnbringer-32",
        author: "DawnBringer",
        colors: &[
            "#000000", "#222034", "#45283c", "#663931", "#8f563b", "#df7126", "#d9a066", "#eec39a", "#fbf236", "#99e550", "#6abe30", "#37946e",
            "#4b692f", "#524b24", "#323c39", "#3f3f74", "#306082", "#5b6ee1", "#639bff", "#5fcde4", "#cbdbfc", "#ffffff", "#9badb7", "#847e87",
            "#696a6a", "#595652", "#76428a", "#ac3232", "#d95763", "#d77bba", "#8f974a", "#8a6f30",
        ],
    },
    Preset {
        name: "endesga-32",
        author: "Endesga",
        colors: &[
            "#be4a2f", "#d77643", "#ead4aa", "#e4a672", "#b86f50", "#733e39", "#3e2731", "#a22633", "#e43b44", "#f77622", "#feae34", "#fee761",
            "#63c74d", "#3e8948", "#265c42", "#193c3e", "#124e89", "#0099db", "#2ce8f5", "#ffffff", "#c0cbdc", "#8b9bb4", "#5a6988", "#3a4466",
            "#262b44", "#181425", "#ff0044", "#68386c", "#b55088", "#f6757a", "#e8b796", "#c28569",
        ],
    },
    Preset {
        name: "slso8",
        author: "Luis Miguel Maldonado",
        colors: &["#0d2b45", "#203c56", "#544e68", "#8d697a", "#d08159", "#ffaa5e", "#ffd4a3", "#ffecd6"],
    },
    Preset {
        name: "gameboy",
        author: "Nintendo Game Boy (DMG) greens, the common approximation",
        colors: &["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    },
];

pub fn find(name: &str) -> Option<&'static Preset> {
    let key = |s: &str| s.chars().filter(char::is_ascii_alphanumeric).collect::<String>().to_ascii_lowercase();
    let k = key(name);
    PRESETS.iter().find(|p| key(p.name) == k || (k == "db16" && p.name == "dawnbringer-16") || (k == "db32" && p.name == "dawnbringer-32"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_sizes_and_hex() {
        let sizes: Vec<(&str, usize)> = PRESETS.iter().map(|p| (p.name, p.colors.len())).collect();
        assert_eq!(sizes, [("pico-8", 16), ("sweetie-16", 16), ("dawnbringer-16", 16), ("dawnbringer-32", 32), ("endesga-32", 32), ("slso8", 8), ("gameboy", 4)]);
        for p in PRESETS {
            for c in p.colors {
                assert!(crate::palette_ops::parse_color_code(c).is_some(), "{} {c}", p.name);
                assert_eq!(c.len(), 7);
            }
            let mut uniq: Vec<_> = p.colors.to_vec();
            uniq.sort();
            uniq.dedup();
            assert_eq!(uniq.len(), p.colors.len(), "{} has duplicates", p.name);
        }
        assert_eq!(find("DB32").unwrap().name, "dawnbringer-32");
        assert_eq!(find("Pico 8").unwrap().name, "pico-8");
    }
}
