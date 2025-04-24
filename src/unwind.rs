use super::*;

pub type DropFn = unsafe extern "C-unwind" fn(*mut ());

#[cfg(feature = "unwind")]
pub type DropInPlaceUnwind = Option<DropFn>;

#[cfg(feature = "unwind")]
pub mod drops {
    use super::*;
    pub unsafe extern "C-unwind" fn do_drop<T>(data: *mut ()) {
        super::do_drop_impl::<T>(data)
    }
    pub unsafe extern "C-unwind" fn do_dyn_drop<T: DropFunc>(data: *mut ()) {
        super::do_dyn_drop_impl::<T>(data)
    }
}

#[cfg(feature = "unwind")]
impl DropFunc for DropInPlaceUnwind {
    unsafe fn invoke(self, data: *mut ()) {
        if let Some(drop_fn) = self {
            drop_fn(data);
        }
    }
}
