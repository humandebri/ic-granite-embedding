// The package name unconditionally selects the original F32 storage format.
const COMPACT_VOCABULARY: bool = false;
include!("canister.rs");

#[cfg(test)]
mod package_tests {
    #[test]
    fn original_package_always_selects_f32_storage() {
        assert_eq!(super::MAGIC, b"BEK25001");
        assert_eq!(super::PACK_SHA256, bekko25_core::PACK_SHA256);
        assert_eq!(super::EMBEDDING_SPACE, bekko25_core::EMBEDDING_SPACE);
    }
}
