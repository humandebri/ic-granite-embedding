//! Export the same Candid declaration without compiling inference kernels in Wasmtime.
fn main() {
    let pointer = granite_canister::get_candid_pointer();
    // get_candid_pointer creates this CString with into_raw; take ownership once.
    let declaration = unsafe { std::ffi::CString::from_raw(pointer) };
    println!("{}", declaration.to_str().unwrap());
}
