//! Verify both model APIs and relocated pack metadata against the pinned pack.
use bekko25_core::{ExternalModel, Model, PACK_BYTES, PACK_SHA256, VOCAB};
use sha2::{Digest, Sha256};
use std::fs;

fn read_pack(pack: &[u8], offset: usize, dest: &mut [u8]) -> Result<(), String> {
    let end = offset.checked_add(dest.len()).ok_or("read overflow")?;
    dest.copy_from_slice(pack.get(offset..end).ok_or("read out of bounds")?);
    Ok(())
}

// Insert whitespace into the JSON header so all tensor offsets shift without
// changing their relative descriptors or allocating a second complete pack.
fn read_shifted(pack: &[u8], padding: usize, offset: usize, dest: &mut [u8]) -> Result<(), String> {
    let metadata_len = usize::try_from(u64::from_le_bytes(pack[..8].try_into().unwrap()))
        .map_err(|_| "metadata overflow")?;
    let old_data = 8 + metadata_len;
    let new_data = old_data.checked_add(padding).ok_or("metadata overflow")?;
    let new_size = pack.len().checked_add(padding).ok_or("pack overflow")?;
    if offset
        .checked_add(dest.len())
        .is_none_or(|end| end > new_size)
    {
        return Err("read out of bounds".into());
    }
    let new_header = ((metadata_len + padding) as u64).to_le_bytes();
    let mut at = offset;
    let mut remaining = dest;
    while !remaining.is_empty() {
        let (take, source) = if at < 8 {
            ((8 - at).min(remaining.len()), Some((&new_header[..], at)))
        } else if at < old_data {
            ((old_data - at).min(remaining.len()), Some((pack, at)))
        } else if at < new_data {
            ((new_data - at).min(remaining.len()), None)
        } else {
            (remaining.len(), Some((pack, at - padding)))
        };
        let (chunk, tail) = remaining.split_at_mut(take);
        if let Some((source, source_at)) = source {
            chunk.copy_from_slice(&source[source_at..source_at + take]);
        } else {
            chunk.fill(b' ');
        }
        at += take;
        remaining = tail;
    }
    Ok(())
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.iter()
        .map(|x| x.to_bits())
        .eq(b.iter().map(|x| x.to_bits()))
}

fn main() {
    let pack = fs::read(".cache/bekko25/model.pack").expect("pinned model pack");
    assert_eq!(pack.len(), PACK_BYTES);
    assert_eq!(format!("{:x}", Sha256::digest(&pack)), PACK_SHA256);
    let ids = [2, VOCAB - 1, 2, 1];

    let resident = Model::load_from(pack.len(), |offset, dest| read_pack(&pack, offset, dest))
        .expect("resident model");
    assert_eq!(
        resident.embed(&ids).unwrap().len(),
        ids.len() * bekko25_core::H
    );
    let expected = resident.full(&ids).unwrap();
    drop(resident);

    let external =
        ExternalModel::load_from(pack.len(), |offset, dest| read_pack(&pack, offset, dest))
            .expect("external model");
    let actual = external
        .full(&ids, |offset, dest| read_pack(&pack, offset, dest))
        .unwrap();
    assert!(bits_equal(&expected, &actual));
    drop(external);

    let padding = 17;
    let shifted = ExternalModel::load_from(pack.len() + padding, |offset, dest| {
        read_shifted(&pack, padding, offset, dest)
    })
    .expect("model with relocated tensor data");
    let actual = shifted
        .full(&ids, |offset, dest| {
            read_shifted(&pack, padding, offset, dest)
        })
        .unwrap();
    assert!(
        bits_equal(&expected, &actual),
        "relocated metadata changed output"
    );
    println!("resident, external, and relocated-metadata outputs match bit-for-bit");
}
