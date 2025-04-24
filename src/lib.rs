//! # tarc
//!
//! ## Transposable, type-erasable, and FFI-compatible Arc.
//!
//! `tarc` is a fork of `std::sync::Arc` with several additions easing its use in type-erased
//! scenarios.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc};
use core::alloc::Layout;
use core::marker::Unpin;
use core::pin::Pin;
#[cfg(not(feature = "std"))]
use core::prelude::rust_2018::*;
use core::ptr::NonNull;
use core::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc};

#[cfg(feature = "unwind")]
mod unwind;
#[cfg(feature = "unwind")]
use unwind::*;

#[cfg(not(feature = "unwind"))]
type DropFn = unsafe extern "C" fn(*mut ());

/// An opaque handle.
#[repr(transparent)]
pub struct Handle(BaseArc<()>);

impl<T: ?Sized> From<Arc<T>> for Handle {
    fn from(arc: Arc<T>) -> Self {
        Self(unsafe { arc.untranspose::<()>() })
    }
}

impl<T: ?Sized> From<BaseArc<T>> for Handle {
    fn from(arc: BaseArc<T>) -> Self {
        let data = arc.into_raw();
        Self(unsafe { BaseArc::from_raw(data as *const ()) })
    }
}

#[repr(C)]
struct ArcHeader {
    count: AtomicUsize,
    drop: DropFn,
}

pub type DropInPlace = Option<unsafe extern "C" fn(*mut ())>;

#[cfg(not(feature = "unwind"))]
pub type DropInPlaceDefault = DropInPlace;
#[cfg(feature = "unwind")]
pub type DropInPlaceDefault = DropInPlaceUnwind;

mod sealed {
    use super::*;

    pub trait DropFunc: Copy {
        unsafe fn invoke(self, data: *mut ());
    }

    impl DropFunc for DropInPlace {
        unsafe fn invoke(self, data: *mut ()) {
            if let Some(drop_fn) = self {
                drop_fn(data);
            }
        }
    }
}

use sealed::DropFunc;

#[repr(C)]
struct DynArcHeader<T> {
    size: usize,
    alignment: usize,
    drop_in_place: T,
    hdr: ArcHeader,
}

unsafe fn do_drop_impl<T>(data: *mut ()) {
    core::ptr::drop_in_place(data as *mut T);
    let (header, layout) = layout::<T>();
    dealloc((data as *mut u8).sub(header), layout);
}

unsafe fn do_dyn_drop_impl<T: DropFunc>(data: *mut ()) {
    let header = (data as *mut DynArcHeader<T>).sub(1);
    let layout = Layout::from_size_align_unchecked((*header).size, (*header).alignment);

    (*header).drop_in_place.invoke(data);

    dealloc(header as *mut u8, layout);
}

#[cfg(not(feature = "unwind"))]
mod drops {
    use super::*;
    pub unsafe extern "C" fn do_drop<T>(data: *mut ()) {
        super::do_drop_impl::<T>(data)
    }
    pub unsafe extern "C" fn do_dyn_drop<T: DropFunc>(data: *mut ()) {
        super::do_dyn_drop_impl::<T>(data)
    }
}

use drops::*;

fn layout<T>() -> (usize, Layout) {
    header_layout::<DynArcHeader<DropInPlaceDefault>>(unsafe {
        Layout::from_size_align_unchecked(core::mem::size_of::<T>(), core::mem::align_of::<T>())
    })
}

fn header_layout<H>(inp: Layout) -> (usize, Layout) {
    let alignment = core::cmp::max(inp.align(), core::mem::align_of::<H>());
    let header_size = core::mem::size_of::<H>();

    // Alignment is always a power of two, thus bitwise ops are valid
    let align_up = |val| (val + (alignment - 1)) & !(alignment - 1);

    let header_aligned = align_up(header_size);
    let size = header_aligned + align_up(inp.size());

    // SAFETY: left half and right half of the size calculation are multiple of alignment
    let layout = unsafe { Layout::from_size_align_unchecked(size, alignment) };

    (header_aligned, layout)
}

unsafe fn allocate<T>() -> NonNull<T> {
    let (header, layout) = layout::<T>();
    let data = alloc(layout).add(header);

    assert!(!data.is_null());

    NonNull::new_unchecked(data as *mut T)
}

unsafe fn initialize<T>(data: NonNull<T>, val: T) {
    let header = (data.as_ptr() as *mut ArcHeader).sub(1);
    header.write(ArcHeader {
        count: AtomicUsize::new(1),
        drop: do_drop::<T>,
    });
    data.as_ptr().write(val);
}

/// Arc that is safe to transpose and type-erase.
///
/// Standard `BaseArc<T>` allocates space for a header and places the header contents at the start
/// of that space, which means the offset between data and header depends on the type alignment.
/// This Arc puts the header right next to the data, making header accesses not rely on the
/// underlying type. This allows to safely transmute `Arc<T>` into `Arc<()>`, or any type that `T`
/// is safe to transmute into.
///
/// In addition, you may convert `BaseArc<T>` to `Arc<O>` if `O` is stored within `T` and there is
/// a `AsRef<O>` implementation on `T`.
///
/// Semantically, this arc differs from the standard library in its absence of weak references.
/// This is so that the header fits within 2 pointers.
#[repr(transparent)]
pub struct BaseArc<T: ?Sized> {
    data: NonNull<T>,
}

impl BaseArc<()> {
    /// Create a custom Arc.
    ///
    /// This function allows to create a custom dynamically sized arc with custom cleanup routine.
    ///
    /// Note that the contents of the created arc are uninitialized.
    ///
    /// # Arguments
    ///
    /// - `layout` - size and alignment of the object being created.
    /// - `drop_in_place` - optional cleanup routine to be invoked on upon releasing the arc.
    ///
    /// # Panics
    ///
    /// This function panics if it fails to allocate enough data for the arc.
    ///
    /// # Safety
    ///
    /// Technically only correct `drop_in_place` implementation needs to be provided. That means,
    /// the caller must ensure that `drop_in_place` releases the contents of the initialized arc
    /// (after it has been returned from this function to the caller), correctly. If `None` is
    /// passed as the cleanup routine, this function should be safe.
    pub unsafe fn custom<T: DropFunc>(layout: Layout, drop_in_place: T) -> Self {
        let (header, layout) = header_layout::<DynArcHeader<T>>(layout);
        let data = alloc(layout);

        assert!(!data.is_null());

        let hdr = data as *mut DynArcHeader<T>;

        hdr.write(DynArcHeader {
            size: layout.size(),
            alignment: layout.align(),
            drop_in_place,
            hdr: ArcHeader {
                count: AtomicUsize::new(1),
                drop: do_dyn_drop::<T>,
            },
        });

        let data = NonNull::new_unchecked(data.add(header) as *mut _);

        Self { data }
    }
}

impl<T> BaseArc<T> {
    pub fn new(val: T) -> Self {
        let data = unsafe { allocate::<T>() };
        unsafe {
            initialize::<T>(data, val);
        }

        Self { data }
    }

    pub fn pin(data: T) -> Pin<Self> {
        unsafe { Pin::new_unchecked(Self::new(data)) }
    }

    pub fn transpose<O>(self) -> Arc<O>
    where
        T: AsRef<O>,
    {
        // Make sure the type can at least be within T
        assert!(core::mem::size_of::<O>() <= core::mem::size_of::<T>());

        let transposed = (*self).as_ref() as *const _ as *const u8;
        let raw = self.into_raw() as *const u8;

        unsafe {
            let off = (transposed as usize).checked_sub(raw as usize).unwrap();

            // Make sure the pointer is within T
            assert!(off <= (core::mem::size_of::<T>() - core::mem::size_of::<O>()));

            Arc::from_raw(raw.add(off) as *const O, off)
        }
    }
}

impl<T: ?Sized> BaseArc<T> {
    fn header(&self) -> &ArcHeader {
        unsafe { &*((self.data.as_ptr() as *mut u8) as *mut ArcHeader).sub(1) }
    }

    /// Convert a pointer to managed `BaseArc`.
    ///
    /// # Safety
    ///
    /// `data` must be originally created through `BaseArc::new`.
    pub unsafe fn from_raw(data: *const T) -> Self {
        Self {
            data: NonNull::new_unchecked(data as *mut _),
        }
    }

    pub fn into_raw(self) -> *const T {
        let ret = self.data.as_ptr() as *const _;
        core::mem::forget(self);
        ret
    }

    /// Increment the strong reference count
    ///
    /// This function takes a pointer to `BaseArc`-allocated data and increments its reference
    /// count.
    ///
    /// # Safety
    ///
    /// `ptr` must be originally created by `BaseArc::new`.
    pub unsafe fn increment_strong_count(ptr: *const T) {
        // Retain Arc, but don't touch refcount by wrapping in ManuallyDrop
        let arc = core::mem::ManuallyDrop::new(Self::from_raw(ptr));
        // Now increase refcount, but don't drop new refcount either
        let _arc_clone: core::mem::ManuallyDrop<_> = arc.clone();
    }

    /// Decrement the strong reference count
    ///
    /// This function takes a pointer to `BaseArc`-allocated data and decrements its reference
    /// count. If the count reaches 0, destructor is called.
    ///
    /// # Safety
    ///
    /// `ptr` must be originally created by `BaseArc::new`.
    pub unsafe fn decrement_strong_count(ptr: *const T) {
        core::mem::drop(BaseArc::from_raw(ptr));
    }

    // Needed for macro drop impl
    unsafe fn as_original_ptr<O: Sized>(&self) -> *const O {
        (self.data.as_ptr() as *const u8) as *const O
    }
}

/// Arc that is safe to transpose and type-erase.
///
/// Standard `Arc<T>` allocates space for a header and places the header contents at the start of
/// that space, which means the offset between data and header depends on the type alignment. This
/// Arc puts the header right next to the data, making header accesses not rely on the underlying
/// type. This allows to safely transmute `Arc<T>` into `Arc<()>`, or any type that `T` is safe to
/// transmute into.
///
/// In addition, you may convert `Arc<T>` to `Arc<O>` if `O` is stored within `T` and there is a
/// `AsRef<O>` implementation on `T`.
///
/// Semantically, this arc also differs from the standard library in its absence of weak references.
/// This is so that the header fits within 2 pointers.
#[repr(C)]
pub struct Arc<T: ?Sized> {
    data: NonNull<T>,
    offset: usize,
}

impl<T> Arc<T> {
    pub fn new(val: T) -> Self {
        let data = unsafe { allocate::<T>() };
        unsafe {
            initialize::<T>(data, val);
        }

        Self { data, offset: 0 }
    }

    pub fn pin(data: T) -> Pin<Self> {
        unsafe { Pin::new_unchecked(Self::new(data)) }
    }

    pub fn transpose<O>(self) -> Arc<O>
    where
        T: AsRef<O>,
    {
        // Make sure the type can at least be within T
        assert!(core::mem::size_of::<O>() <= core::mem::size_of::<T>());

        let transposed = (*self).as_ref() as *const _ as *const u8;
        let (raw, offset) = self.into_raw();
        let raw = raw as *const u8;

        unsafe {
            let off = (transposed as usize).checked_sub(raw as usize).unwrap();

            // Make sure the pointer is within T
            assert!(off <= (core::mem::size_of::<T>() - core::mem::size_of::<O>()));

            Arc::from_raw(raw.add(off) as *const O, offset + off)
        }
    }
}

impl<T: ?Sized> Arc<T> {
    fn header(&self) -> &ArcHeader {
        unsafe { &*((self.data.as_ptr() as *mut u8).sub(self.offset) as *mut ArcHeader).sub(1) }
    }

    /// Convert raw pointer to a managed `Arc`.
    ///
    /// # Safety
    ///
    /// `ptr` and `offset` needs to be a pair acquired from `Arc::into_raw` function call. More
    /// specifically, `ptr as *const u8 - offset` must point to start of the data of original
    /// `BaseArc`.
    pub unsafe fn from_raw(data: *const T, offset: usize) -> Self {
        Self {
            data: NonNull::new_unchecked(data as *mut _),
            offset,
        }
    }

    pub fn into_raw(self) -> (*const T, usize) {
        let ret = self.data.as_ptr() as *const _;
        let offset = self.offset;
        core::mem::forget(self);
        (ret, offset)
    }

    /// Increment the strong reference count.
    ///
    /// Takes pointer to Arc's data and byte offset of the data and increments the strong reference
    /// count.
    ///
    /// # Safety
    ///
    /// `ptr` and `offset` needs to be a pair acquired from `Arc::into_raw` function call. More
    /// specifically, `ptr as *const u8 - offset` must point to start of the data of original
    /// `BaseArc`.
    pub unsafe fn increment_strong_count(ptr: *const T, offset: usize) {
        // Retain Arc, but don't touch refcount by wrapping in ManuallyDrop
        let arc = core::mem::ManuallyDrop::new(Self::from_raw(ptr, offset));
        // Now increase refcount, but don't drop new refcount either
        let _arc_clone: core::mem::ManuallyDrop<_> = arc.clone();
    }

    /// Decrement the strong reference count.
    ///
    /// Takes pointer to Arc's data and byte offset of the data and decrements the strong reference
    /// count.
    ///
    /// If the reference count reaches 0, destructor is called.
    ///
    /// # Safety
    ///
    /// `ptr` and `offset` needs to be a pair acquired from `Arc::into_raw` function call. More
    /// specifically, `ptr as *const u8 - offset` must point to start of the data of original
    /// `BaseArc`.
    pub unsafe fn decrement_strong_count(ptr: *const T, offset: usize) {
        core::mem::drop(Arc::from_raw(ptr, offset));
    }

    /// Get `BaseArc<O>` out of this `Arc`
    ///
    /// # Safety
    ///
    /// The base arc must be of type `O`.
    pub unsafe fn untranspose<O>(self) -> BaseArc<O> {
        let (raw, offset) = self.into_raw();
        let raw = (raw as *const u8).sub(offset) as *const O;
        BaseArc::from_raw(raw)
    }

    /// Get the pointer to data of `BaseArc<O>`.
    ///
    /// # Safety
    ///
    /// The base arc must be of type `O`.
    pub unsafe fn as_original_ptr<O>(&self) -> *const O {
        (self.data.as_ptr() as *const u8).sub(self.offset) as *const O
    }

    pub fn is_original(&self) -> bool {
        self.offset == 0
    }

    pub fn into_base(self) -> Result<BaseArc<T>, Self> {
        if self.offset == 0 {
            let (raw, _) = self.into_raw();
            Ok(unsafe { BaseArc::from_raw(raw) })
        } else {
            Err(self)
        }
    }
}

/// Expand all function-like macros in docs before docs themselves are interpreted.
///
/// This is a workaround for rustc < 1.54, where doc attributes cannot have functional macros in
/// them.
macro_rules! doc {
    ( $(#[doc = $expr1:expr])* ### #[doc = $expr2:expr] $($tt:tt)* ) => {
        doc! {
            $(#[doc = $expr1])*
            #[doc = $expr2]
            ###
            $($tt)*
        }
    };
    ( $(#[doc = $exprs:expr])* ### $($tt:tt)* ) => {
        $(#[doc = $exprs])*
        $($tt)*
    };
    ( $($tt:tt)* ) => {
        doc! {
            ###
            $($tt)*
        }
    }
}

macro_rules! arc_traits {
    ($mname:ident, $ty:ident) => {

        impl<T: ?Sized> $ty<T> {
            pub fn strong_count(&self) -> usize {
                self.header().count.load(Ordering::Acquire)
            }

            pub fn as_ptr(&self) -> *const T {
                self.data.as_ptr() as *const _
            }

            pub fn exclusive_ptr(&self) -> Option<NonNull<T>> {
                if self.strong_count() == 1 {
                    Some(self.data)
                } else {
                    None
                }
            }

            pub fn get_mut(&mut self) -> Option<&mut T> {
                self.exclusive_ptr().map(|v| unsafe { &mut *v.as_ptr() })
            }
        }

        mod $mname {
            use super::*;
            impl<T: Default> Default for $ty<T> {
                doc! {
                    #[doc = concat!("Creates a new `", stringify!($ty), "<T>`, with the `Default` value for `T`.")]
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    #[doc = concat!("let x: ", stringify!($ty), "<i32> = Default::default();")]
                    /// assert_eq!(*x, 0);
                    /// ```
                    fn default() -> Self {
                        Self::new(T::default())
                    }
                }
            }

            impl<T: ?Sized> core::ops::Deref for $ty<T> {
                type Target = T;

                fn deref(&self) -> &Self::Target {
                    unsafe { self.data.as_ref() }
                }
            }

            impl<T: ?Sized + core::hash::Hash> core::hash::Hash for $ty<T> {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    (**self).hash(state)
                }
            }

            impl<T: ?Sized> Unpin for $ty<T> {}

            unsafe impl<T: ?Sized + Sync + Send> Send for $ty<T> {}
            unsafe impl<T: ?Sized + Sync + Send> Sync for $ty<T> {}

            doc! {
                #[doc = concat!("Converts a `T` into an `", stringify!($ty), "<T>`")]
                ///
                /// The conversion moves the value into a
                #[doc = concat!("newly allocated `", stringify!($ty), "`. It is equivalent to")]
                #[doc = concat!(" calling `", stringify!($ty), "::new(t)`")]
                ///
                /// # Example
                /// ```rust
                #[doc = concat!("# use tarc:: ", stringify!($ty), ";")]
                /// let x = 5;
                #[doc = concat!("let arc = ", stringify!($ty), "::new(5);")]
                ///
                #[doc = concat!("assert_eq!(", stringify!($ty), "::from(x), arc);")]
                /// ```
                impl<T> From<T> for $ty<T> {
                    fn from(t: T) -> Self {
                        Self::new(t)
                    }
                }
            }

            impl<T: ?Sized> Clone for $ty<T> {
                doc! {
                    #[doc = concat!("Makes a clone of the `", stringify!($ty), "` pointer.")]
                    ///
                    /// This creates another pointer to the same allocation, increasing the
                    /// strong reference count.
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("let _ = ", stringify!($ty), "::clone(&five);")]
                    /// ```
                    fn clone(&self) -> Self {
                        if self.header().count.fetch_add(1, Ordering::Relaxed) > core::isize::MAX as usize {
                            panic!()
                        }

                        Self { ..*self }
                    }
                }
            }

            impl<T: ?Sized> Drop for $ty<T> {
                doc! {
                    #[doc = concat!("Drops the `", stringify!($ty), "`.")]
                    ///
                    /// This will decrement the reference count. If the strong reference
                    /// count reaches zero then the no more references, so we `drop` the inner value.
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    /// struct Foo;
                    ///
                    /// impl Drop for Foo {
                    ///     fn drop(&mut self) {
                    ///         println!("dropped!");
                    ///     }
                    /// }
                    ///
                    #[doc = concat!("let foo  = ", stringify!($ty), "::new(Foo);")]
                    #[doc = concat!("let foo2 = ", stringify!($ty), "::clone(&foo);")]
                    ///
                    /// drop(foo);    // Doesn't print anything
                    /// drop(foo2);   // Prints "dropped!"
                    /// ```
                    fn drop(&mut self) {
                        let header = self.header();

                        if header.count.fetch_sub(1, Ordering::Release) != 1 {
                            return;
                        }

                        fence(Ordering::SeqCst);

                        unsafe { (header.drop)(self.as_original_ptr::<()>() as *mut ()) }
                    }
                }
            }

            impl<T: ?Sized + PartialEq> PartialEq for $ty<T> {
                doc! {
                    #[doc = concat!("Equality for two `", stringify!($ty), "`s.")]
                    ///
                    #[doc = concat!("Two `", stringify!($ty), "`s are equal if their inner values are equal, even if they are")]
                    /// stored in different allocation.
                    ///
                    /// If `T` also implements `Eq` (implying reflexivity of equality),
                    #[doc = concat!("two `", stringify!($ty), "`s that point to the same allocation are always equal.")]
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("assert!(five == ", stringify!($ty), "::new(5));")]
                    /// ```
                    #[inline]
                    fn eq(&self, other: &$ty<T>) -> bool {
                        // TODO: implement this optimization with specialization
                        /*self.as_ptr() == other.as_ptr() ||*/ (**self == **other)
                    }
                }

                doc! {
                    #[doc = concat!("Inequality for two `", stringify!($ty), "`s.")]
                    ///
                    #[doc = concat!("Two `", stringify!($ty), "`s are unequal if their inner values are unequal.")]
                    ///
                    /// If `T` also implements `Eq` (implying reflexivity of equality),
                    #[doc = concat!("two `", stringify!($ty), "`s that point to the same value are never unequal.")]
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("assert!(five != ", stringify!($ty), "::new(6));")]
                    /// ```
                    #[inline]
                    #[allow(clippy::partialeq_ne_impl)]
                    fn ne(&self, other: &$ty<T>) -> bool {
                        // TODO: implement this optimization with specialization
                        /*self.as_ptr() != other.as_ptr() &&*/ (**self != **other)
                    }
                }
            }

            use core::cmp::{Ord, Ordering as CmpOrdering};

            impl<T: ?Sized + PartialOrd> PartialOrd for $ty<T> {
                doc! {
                    #[doc = concat!("Partial comparison for two `", stringify!($ty), "`s.")]
                    ///
                    /// The two are compared by calling `partial_cmp()` on their inner values.
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    /// use std::cmp::Ordering;
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("assert_eq!(Some(Ordering::Less), five.partial_cmp(&", stringify!($ty), "::new(6)));")]
                    /// ```
                    fn partial_cmp(&self, other: &$ty<T>) -> Option<CmpOrdering> {
                        (**self).partial_cmp(&**other)
                    }
                }

                doc! {
                    #[doc = concat!("Less-than comparison for two `", stringify!($ty), "`s.")]
                    ///
                    /// The two are compared by calling `<` on their inner values.
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("assert!(five < ", stringify!($ty), "::new(6));")]
                    /// ```
                    fn lt(&self, other: &$ty<T>) -> bool {
                        *(*self) < *(*other)
                    }
                }

                doc! {
                    #[doc = concat!("'Less than or equal to' comparison for two `", stringify!($ty), "`s.")]
                    ///
                    /// The two are compared by calling `<=` on their inner values.
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("assert!(five <= ", stringify!($ty), "::new(5));")]
                    /// ```
                    fn le(&self, other: &$ty<T>) -> bool {
                        *(*self) <= *(*other)
                    }
                }

                doc! {
                    #[doc = concat!("Greater-than comparison for two `", stringify!($ty), "`s.")]
                    ///
                    /// The two are compared by calling `>` on their inner values.
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("assert!(five > ", stringify!($ty), "::new(4));")]
                    /// ```
                    fn gt(&self, other: &$ty<T>) -> bool {
                        *(*self) > *(*other)
                    }
                }

                doc! {
                    #[doc = concat!("'Greater than or equal to' comparison for two `", stringify!($ty), "`s.")]
                    ///
                    /// The two are compared by calling `>=` on their inner values.
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("assert!(five >= ", stringify!($ty), "::new(5));")]
                    /// ```
                    fn ge(&self, other: &$ty<T>) -> bool {
                        *(*self) >= *(*other)
                    }
                }
            }

            impl<T: ?Sized + Eq> Eq for $ty<T> {}

            impl<T: ?Sized + Ord> Ord for $ty<T> {
                doc! {
                    #[doc = concat!("Comparison for two `", stringify!($ty), "`s.")]
                    ///
                    /// The two are compared by calling `cmp()` on their inner values.
                    ///
                    /// # Examples
                    ///
                    /// ```
                    #[doc = concat!("use tarc::", stringify!($ty), ";")]
                    /// use std::cmp::Ordering;
                    ///
                    #[doc = concat!("let five = ", stringify!($ty), "::new(5);")]
                    ///
                    #[doc = concat!("assert_eq!(Ordering::Less, five.cmp(& ", stringify!($ty), "::new(6)));")]
                    /// ```
                    fn cmp(&self, other: &$ty<T>) -> CmpOrdering {
                        (**self).cmp(&**other)
                    }
                }
            }

            use core::fmt;

            impl<T: ?Sized + fmt::Display> fmt::Display for $ty<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmt::Display::fmt(&**self, f)
                }
            }

            impl<T: ?Sized + fmt::Debug> fmt::Debug for $ty<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmt::Debug::fmt(&**self, f)
                }
            }

            impl<T: ?Sized> fmt::Pointer for $ty<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmt::Pointer::fmt(&(&**self as *const T), f)
                }
            }

            impl<T: ?Sized> core::borrow::Borrow<T> for $ty<T> {
                fn borrow(&self) -> &T {
                    &**self
                }
            }

            impl<T: ?Sized> AsRef<T> for $ty<T> {
                fn as_ref(&self) -> &T {
                    &**self
                }
            }

            #[cfg(feature = "std")]
            mod std_impl {
                use super::*;
                use std::error::Error;

                impl<T: Error + ?Sized> Error for $ty<T> {
                    #[allow(deprecated, deprecated_in_future)]
                    fn description(&self) -> &str {
                        Error::description(&**self)
                    }

                    #[allow(deprecated)]
                    fn cause(&self) -> Option<&dyn Error> {
                        Error::cause(&**self)
                    }

                    fn source(&self) -> Option<&(dyn Error + 'static)> {
                        Error::source(&**self)
                    }
                }
            }
        }
    };
}

arc_traits!(base_arc, BaseArc);
arc_traits!(arc, Arc);
