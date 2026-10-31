fn main() {
    let pointer = bekko25_compact_canister::get_candid_pointer();
    let declaration = unsafe { std::ffi::CString::from_raw(pointer) };
    println!("{}", declaration.to_str().unwrap());
}
