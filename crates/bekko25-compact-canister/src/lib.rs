// Share the canister implementation while selecting a distinct storage format.
const COMPACT_VOCABULARY: bool = true;
include!("../../bekko25-canister/src/canister.rs");

#[cfg(test)]
mod package_tests {
    #[test]
    fn compact_package_always_selects_compact_storage() {
        assert_eq!(super::MAGIC, b"BKI25001");
        assert_eq!(super::PACK_SHA256, bekko25_core::COMPACT_PACK_SHA256);
        assert_eq!(
            super::EMBEDDING_SPACE,
            bekko25_core::COMPACT_EMBEDDING_SPACE
        );
    }
}
