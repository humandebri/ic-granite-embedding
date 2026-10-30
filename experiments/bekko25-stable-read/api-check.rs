//! Check the resident and external-row APIs against the real pinned pack.
use bekko25_core::{H, Model, PACK_BYTES, PACK_SHA256};
use sha2::{Digest, Sha256};
use std::fs;

fn main() {
    let pack = fs::read(".cache/bekko25/model.pack").unwrap();
    assert_eq!(pack.len(), PACK_BYTES);
    assert_eq!(format!("{:x}", Sha256::digest(&pack)), PACK_SHA256);
    let read = |offset: usize, dest: &mut [u8]| {
        let end = offset.checked_add(dest.len()).ok_or("read overflow")?;
        let src = pack.get(offset..end).ok_or("read out of bounds")?;
        dest.copy_from_slice(src);
        Ok(())
    };
    let ids = [2, 1];
    let resident = Model::load_from(PACK_BYTES, read).unwrap();
    let expected_embed = resident.embed(&ids).unwrap();
    let expected_full = resident.full(&ids).unwrap();
    drop(resident);

    let external = Model::load_from_external_embeddings(PACK_BYTES, read).unwrap();
    assert_eq!(
        external.embed(&ids).unwrap_err(),
        "model requires external embedding rows"
    );
    assert_eq!(
        external.full(&ids).unwrap_err(),
        "model requires external embedding rows"
    );
    let mut rows = Vec::new();
    for id in ids {
        let start = 8860 + id as usize * H * 4;
        rows.extend(
            pack[start..start + H * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    let actual = external.full_from_rows(&ids, rows).unwrap();
    assert_eq!(expected_embed.len(), ids.len() * H);
    assert!(
        expected_full
            .iter()
            .map(|x| x.to_bits())
            .eq(actual.iter().map(|x| x.to_bits()))
    );
    println!(
        "resident API succeeds, external API returns errors, and external rows match bit-for-bit"
    );
}
