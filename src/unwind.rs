use super::*;

pub type DropFnUnwind = unsafe extern "C-unwind" fn(*mut ());

pub type DropInPlaceUnwind = Option<DropFnUnwind>;

pub mod drops {
    use super::*;
    pub unsafe extern "C-unwind" fn do_drop<T>(data: *mut ()) {
        super::do_drop_impl::<T>(data)
    }
    pub unsafe extern "C-unwind" fn do_dyn_drop<T: DropFunc>(data: *mut ()) {
        super::do_dyn_drop_impl::<T>(data)
    }
}

impl DropFunc for DropInPlaceUnwind {
    unsafe fn invoke(self, data: *mut ()) {
        if let Some(drop_fn) = self {
            drop_fn(data);
        }
    }
}

impl DropFunc for DropFnUnwind {
    unsafe fn invoke(self, data: *mut ()) {
        self(data)
    }
}
