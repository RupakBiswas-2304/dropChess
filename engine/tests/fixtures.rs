//! The Rust engine must reproduce the perft counts written by the Python
//! reference engine (train/make_fixtures.py).

use dropchess::{perft, Position, Variant};

#[test]
fn perft_matches_python_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../train/fixtures/perft.json");
    let text = std::fs::read_to_string(path).expect("train/fixtures/perft.json");
    let entries: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut checked = 0;
    for e in entries.as_array().unwrap() {
        let variant = Variant::parse(e["variant"].as_str().unwrap()).unwrap();
        let pos = Position::from_fen(e["fen"].as_str().unwrap(), variant).unwrap();
        for (depth, n) in e["perft"].as_object().unwrap() {
            let depth: u32 = depth.parse().unwrap();
            assert_eq!(perft(&pos, depth), n.as_u64().unwrap(), "{} {:?} depth {}", e["fen"], variant, depth);
            checked += 1;
        }
    }
    assert!(checked >= 40);
}
