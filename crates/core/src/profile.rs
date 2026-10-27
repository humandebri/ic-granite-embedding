//! Optional local measurement support. Absent from the production build.
use serde::Serialize;
use std::{cell::RefCell, collections::BTreeMap};

#[derive(Clone, Serialize)]
pub struct Phase {
    pub name: String,
    pub instructions: u64,
    pub calls: u64,
}
#[derive(Default)]
struct State {
    enabled: bool,
    path: Vec<String>,
    phases: BTreeMap<String, (u64, u64)>,
}
thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "ic0")]
unsafe extern "C" {
    fn performance_counter(kind: i32) -> i64;
}
pub fn now() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { performance_counter(0) as u64 }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}
pub fn begin() {
    STATE.with_borrow_mut(|s| {
        *s = State {
            enabled: true,
            ..Default::default()
        }
    });
}
pub fn finish() -> Vec<Phase> {
    STATE.with_borrow_mut(|s| {
        s.enabled = false;
        std::mem::take(&mut s.phases)
            .into_iter()
            .map(|(name, (instructions, calls))| Phase {
                name,
                instructions,
                calls,
            })
            .collect()
    })
}
pub struct Scope {
    start: Option<u64>,
}
pub fn scope(name: &str) -> Scope {
    let enabled = STATE.with_borrow_mut(|s| {
        if s.enabled {
            s.path.push(name.to_string());
        }
        s.enabled
    });
    Scope {
        start: enabled.then(now),
    }
}
pub fn add(name: &str, instructions: u64, calls: u64) {
    STATE.with_borrow_mut(|s| {
        if s.enabled {
            let key = format!("{}/{}", s.path.join("/"), name);
            let e = s.phases.entry(key).or_default();
            e.0 += instructions;
            e.1 += calls;
        }
    });
}
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = now() - start;
            STATE.with_borrow_mut(|s| {
                let key = s.path.join("/");
                let e = s.phases.entry(key).or_default();
                e.0 += elapsed;
                e.1 += 1;
                s.path.pop();
            });
        }
    }
}
