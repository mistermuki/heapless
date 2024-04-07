//! A fixed capacity array backed [`String`](https://doc.rust-lang.org/std/string/struct.String.html).

use core::{
    char::DecodeUtf16Error,
    cmp::Ordering,
    fmt::{self, Arguments, Write},
    hash, iter, ops,
    str::{self, Utf8Error},
};

use crate::Vec;

/// A possible error value when converting a [`ArrayString`] from a UTF-16 byte slice.
///
/// This type is the error type for the [`from_utf16`] method on [`ArrayString`].
///
/// [`from_utf16`]: ArrayString::from_utf16
#[derive(Debug)]
pub enum FromUtf16Error {
    /// The capacity of the `ArrayString` is too small for the given operation.
    Capacity,
    /// Error decoding UTF-16.
    DecodeUtf16Error(DecodeUtf16Error),
}

impl fmt::Display for FromUtf16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity => "insufficient capacity".fmt(f),
            Self::DecodeUtf16Error(e) => write!(f, "invalid UTF-16: {}", e),
        }
    }
}

/// A fixed capacity [`ArrayString`](https://doc.rust-lang.org/std/string/struct.ArrayString.html).
pub struct ArrayString<const N: usize> {
    arr: [u8; N],
    len: usize,
}

impl<const N: usize> Copy for ArrayString<N> {}

impl<const N: usize> ArrayString<N> {
    /// Constructs a new, empty `ArrayString` with a fixed capacity of `N` bytes.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// // allocate the string on the stack
    /// let mut s: ArrayString<4> = ArrayString::new();
    ///
    /// // allocate the string in a static variable
    /// static mut S: ArrayString<4> = ArrayString::new();
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self {
            arr: [0; N],
            len: 0,
        }
    }

    /// Decodes a UTF-16–encoded slice `v` into a `ArrayString`, returning [`Err`]
    /// if `v` contains any invalid data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// // 𝄞music
    /// let v = &[0xD834, 0xDD1E, 0x006d, 0x0075, 0x0073, 0x0069, 0x0063];
    /// let s: ArrayString<14> = ArrayString::from_utf16(v).unwrap();
    /// assert_eq!(s, "𝄞music");
    ///
    /// // 𝄞mu<invalid>ic
    /// let v = &[0xD834, 0xDD1E, 0x006d, 0x0075, 0xD800, 0x0069, 0x0063];
    /// assert!(ArrayString::<14>::from_utf16(v).is_err());
    /// ```
    #[inline]
    pub fn from_utf16(v: &[u16]) -> Result<Self, FromUtf16Error> {
        let mut s = Self::new();

        for c in char::decode_utf16(v.iter().cloned()) {
            match c {
                Ok(c) => {
                    s.push(c).map_err(|_| FromUtf16Error::Capacity)?;
                }
                Err(err) => {
                    return Err(FromUtf16Error::DecodeUtf16Error(err));
                }
            }
        }

        Ok(s)
    }

    /// Convert UTF-8 bytes into a `ArrayString`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::{ArrayString, Vec};
    ///
    /// let mut sparkle_heart = Vec::<u8, 4>::new();
    /// sparkle_heart.extend_from_slice(&[240, 159, 146, 150]);
    ///
    /// let sparkle_heart: ArrayString<4> = ArrayString::from_utf8(sparkle_heart)?;
    /// assert_eq!("💖", sparkle_heart);
    /// # Ok::<(), core::str::Utf8Error>(())
    /// ```
    ///
    /// Invalid UTF-8:
    ///
    /// ```
    /// use core::str::Utf8Error;
    /// use heapless::{ArrayString, Vec};
    ///
    /// let mut vec = Vec::<u8, 4>::new();
    /// vec.extend_from_slice(&[0, 159, 146, 150]);
    ///
    /// let e: Utf8Error = ArrayString::from_utf8(vec).unwrap_err();
    /// assert_eq!(e.valid_up_to(), 1);
    /// # Ok::<(), core::str::Utf8Error>(())
    /// ```
    #[inline]
    pub fn from_utf8(vec: Vec<u8, N>) -> Result<Self, Utf8Error> {
        core::str::from_utf8(&vec)?;
        Ok(Self {
            arr: vec[..].try_into().unwrap(),
            len: vec.len(),
        })
    }

    /// Convert UTF-8 bytes into a `ArrayString`, without checking that the string
    /// contains valid UTF-8.
    ///
    /// # Safety
    ///
    /// The bytes passed in must be valid UTF-8.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::{ArrayString, Vec};
    ///
    /// let mut sparkle_heart = Vec::<u8, 4>::new();
    /// sparkle_heart.extend_from_slice(&[240, 159, 146, 150]);
    ///
    /// // Safety: `sparkle_heart` Vec is known to contain valid UTF-8
    /// let sparkle_heart: ArrayString<4> = unsafe { ArrayString::from_utf8_unchecked(sparkle_heart) };
    /// assert_eq!("💖", sparkle_heart);
    /// ```
    #[inline]
    pub unsafe fn from_utf8_unchecked(vec: Vec<u8, N>) -> Self {
        Self {
            arr: vec[..].try_into().unwrap_unchecked(),
            len: vec.len(),
        }
    }

    /// Converts a `ArrayString` into a byte vector.
    ///
    /// This consumes the `ArrayString`, so we do not need to copy its contents.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let s: ArrayString<4> = ArrayString::try_from("ab")?;
    /// let b = s.into_bytes();
    /// assert!(b.len() == 2);
    ///
    /// assert_eq!(&[b'a', b'b'], &b[..]);
    /// # Ok::<(), ()>(())
    /// ```
    #[inline]
    pub fn into_bytes(self) -> Vec<u8, N> {
        self.arr[0..self.len].try_into().unwrap()
    }

    /// Extracts a string slice containing the entire string.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let mut s: ArrayString<4> = ArrayString::try_from("ab")?;
    /// assert!(s.as_str() == "ab");
    ///
    /// let _s = s.as_str();
    /// // s.push('c'); // <- cannot borrow `s` as mutable because it is also borrowed as immutable
    /// # Ok::<(), ()>(())
    /// ```
    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe { str::from_utf8_unchecked(&self.arr[0..self.len]) }
    }

    /// Converts a `ArrayString` into a mutable string slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let mut s: ArrayString<4> = ArrayString::try_from("ab")?;
    /// let s = s.as_mut_str();
    /// s.make_ascii_uppercase();
    /// # Ok::<(), ()>(())
    /// ```
    #[inline]
    pub fn as_mut_str(&mut self) -> &mut str {
        unsafe { str::from_utf8_unchecked_mut(&mut self.arr[0..self.len]) }
    }

    /// Appends a given string slice onto the end of this `ArrayString`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let mut s: ArrayString<8> = ArrayString::try_from("foo")?;
    ///
    /// assert!(s.push_str("bar").is_ok());
    ///
    /// assert_eq!("foobar", s);
    ///
    /// assert!(s.push_str("tender").is_err());
    /// # Ok::<(), ()>(())
    /// ```
    #[inline]
    #[allow(clippy::result_unit_err)]
    pub fn push_str(&mut self, string: &str) -> Result<(), ()> {
        let string_length = string.len();
        // If we are going to overflow.
        if self.len + string_length > N {
            return Err(())
        }

        for (i, c) in string.as_bytes().iter().enumerate() {
            self.arr[i + self.len] = *c; 
        }
        self.len += string_length; 

        Ok(())
    }

    /// Returns the maximum number of elements the ArrayString can hold.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let mut s: ArrayString<4> = ArrayString::new();
    /// assert!(s.capacity() == 4);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Appends the given [`char`] to the end of this `ArrayString`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let mut s: ArrayString<8> = ArrayString::try_from("abc")?;
    ///
    /// s.push('1').unwrap();
    /// s.push('2').unwrap();
    /// s.push('3').unwrap();
    ///
    /// assert!("abc123" == s.as_str());
    ///
    /// assert_eq!("abc123", s);
    /// # Ok::<(), ()>(())
    /// ```
    #[inline]
    #[allow(clippy::result_unit_err)]
    pub fn push(&mut self, c: char) -> Result<(), ()> {
        let char_length = c.len_utf8();

        if self.len + char_length > N {
            return Err(());
        }

        match char_length {
            1 => {
                self.arr[self.len] = c as u8;
                self.len += 1;
                Ok(())
            }
            _ => {
                self.push_str(c.encode_utf8(&mut [0; 4]))?;
                Ok(())
            }
        }
    }

    /// Shortens this `ArrayString` to the specified length.
    ///
    /// If `new_len` is greater than the string's current length, this has no
    /// effect.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the string
    ///
    /// # Panics
    ///
    /// Panics if `new_len` does not lie on a [`char`] boundary.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let mut s: ArrayString<8> = ArrayString::try_from("hello")?;
    ///
    /// s.truncate(2);
    ///
    /// assert_eq!("he", s);
    /// # Ok::<(), ()>(())
    /// ```
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        if new_len <= self.len() {
            assert!(self.is_char_boundary(new_len));

            // Set all after the new_len to 0.
            for i in 0..(self.len - new_len) {
                self.arr[i + new_len] = 0;
            }

            self.len = new_len
        }
    }

    /// Removes the last character from the string buffer and returns it.
    ///
    /// Returns [`None`] if this `ArrayString` is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let mut s: ArrayString<8> = ArrayString::try_from("foo")?;
    ///
    /// assert_eq!(s.pop(), Some('o'));
    /// assert_eq!(s.pop(), Some('o'));
    /// assert_eq!(s.pop(), Some('f'));
    ///
    /// assert_eq!(s.pop(), None);
    /// Ok::<(), ()>(())
    /// ```
    pub fn pop(&mut self) -> Option<char> {
        if self.len == 0 {
            return None;
        }

        // pop bytes that correspond to `ch`
        let ch = self.chars().next_back()?;
        let char_length = ch.len_utf8();
        for i in 0..char_length {
            self.arr[self.len - i] = 0;
        }
        self.len -= char_length;

        Some(ch)
    }

    /// Truncates this `ArrayString`, removing all contents.
    ///
    /// While this means the `ArrayString` will have a length of zero, it does not
    /// touch its capacity.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use heapless::ArrayString;
    ///
    /// let mut s: ArrayString<8> = ArrayString::try_from("foo")?;
    ///
    /// s.clear();
    ///
    /// assert!(s.is_empty());
    /// assert_eq!(0, s.len());
    /// assert_eq!(8, s.capacity());
    /// Ok::<(), ()>(())
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        for i in 0..N {
            self.arr[i] = 0;
        }

        self.len = 0;
    }
}

impl<const N: usize> Default for ArrayString<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, const N: usize> TryFrom<&'a str> for ArrayString<N> {
    type Error = ();
    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        let mut new = ArrayString::new();
        new.push_str(s)?;
        Ok(new)
    }
}

impl<const N: usize> str::FromStr for ArrayString<N> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut new = ArrayString::new();
        new.push_str(s)?;
        Ok(new)
    }
}

impl<const N: usize> iter::FromIterator<char> for ArrayString<N> {
    fn from_iter<T: IntoIterator<Item = char>>(iter: T) -> Self {
        let mut new = ArrayString::new();
        for c in iter {
            new.push(c).unwrap();
        }
        new
    }
}

impl<'a, const N: usize> iter::FromIterator<&'a char> for ArrayString<N> {
    fn from_iter<T: IntoIterator<Item = &'a char>>(iter: T) -> Self {
        let mut new = ArrayString::new();
        for c in iter {
            new.push(*c).unwrap();
        }
        new
    }
}

impl<'a, const N: usize> iter::FromIterator<&'a str> for ArrayString<N> {
    fn from_iter<T: IntoIterator<Item = &'a str>>(iter: T) -> Self {
        let mut new = ArrayString::new();
        for c in iter {
            new.push_str(c).unwrap();
        }
        new
    }
}

impl<const N: usize> Clone for ArrayString<N> {
    fn clone(&self) -> Self {
        Self {
            arr: self.arr.clone(),
            len: self.len.clone(),
        }
    }
}

impl<const N: usize> fmt::Debug for ArrayString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <str as fmt::Debug>::fmt(self, f)
    }
}

impl<const N: usize> fmt::Display for ArrayString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <str as fmt::Display>::fmt(self, f)
    }
}

impl<const N: usize> hash::Hash for ArrayString<N> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        <str as hash::Hash>::hash(self, hasher)
    }
}

impl<const N: usize> fmt::Write for ArrayString<N> {
    fn write_str(&mut self, s: &str) -> Result<(), fmt::Error> {
        self.push_str(s).map_err(|_| fmt::Error)
    }

    fn write_char(&mut self, c: char) -> Result<(), fmt::Error> {
        self.push(c).map_err(|_| fmt::Error)
    }
}

impl<const N: usize> ops::Deref for ArrayString<N> {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> ops::DerefMut for ArrayString<N> {
    fn deref_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<const N: usize> AsRef<str> for ArrayString<N> {
    #[inline]
    fn as_ref(&self) -> &str {
        self
    }
}

impl<const N: usize> AsRef<[u8]> for ArrayString<N> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<const N1: usize, const N2: usize> PartialEq<ArrayString<N2>> for ArrayString<N1> {
    fn eq(&self, rhs: &ArrayString<N2>) -> bool {
        str::eq(&**self, &**rhs)
    }
}

// ArrayString<N> == str
impl<const N: usize> PartialEq<str> for ArrayString<N> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        str::eq(self, other)
    }
}

// ArrayString<N> == &'str
impl<const N: usize> PartialEq<&str> for ArrayString<N> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        str::eq(self, &other[..])
    }
}

// str == ArrayString<N>
impl<const N: usize> PartialEq<ArrayString<N>> for str {
    #[inline]
    fn eq(&self, other: &ArrayString<N>) -> bool {
        str::eq(self, &other[..])
    }
}

// &'str == ArrayString<N>
impl<const N: usize> PartialEq<ArrayString<N>> for &str {
    #[inline]
    fn eq(&self, other: &ArrayString<N>) -> bool {
        str::eq(self, &other[..])
    }
}

impl<const N: usize> Eq for ArrayString<N> {}

impl<const N1: usize, const N2: usize> PartialOrd<ArrayString<N2>> for ArrayString<N1> {
    #[inline]
    fn partial_cmp(&self, other: &ArrayString<N2>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<const N: usize> Ord for ArrayString<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

/// Equivalent to [`format`](https://doc.rust-lang.org/std/fmt/fn.format.html).
///
/// Please note that using [`format!`] might be preferable.
///
/// # Errors
///
/// There are two possible error cases. Both return the unit type [`core::fmt::Error`].
///
/// - In case the formatting exceeds the string's capacity. This error does not exist in
/// the standard library as the string would just grow.
/// - If a formatting trait implementation returns an error. The standard library panics
/// in this case.
///
/// [`format!`]: crate::format!
#[doc(hidden)]
pub fn format2<const N: usize>(args: Arguments<'_>) -> Result<ArrayString<N>, fmt::Error> {
    fn format_inner<const N: usize>(args: Arguments<'_>) -> Result<ArrayString<N>, fmt::Error> {
        let mut output = ArrayString::new();
        output.write_fmt(args)?;
        Ok(output)
    }

    args.as_str().map_or_else(
        || format_inner(args),
        |s| s.try_into().map_err(|_| fmt::Error),
    )
}

/// Macro that creates a fixed capacity [`ArrayString`]. Equivalent to [`format!`](https://doc.rust-lang.org/std/macro.format.html).
///
/// The macro's arguments work in the same way as the regular macro.
///
/// It is possible to explicitly specify the capacity of the returned string as the first argument.
/// In this case it is necessary to disambiguate by separating the capacity with a semicolon.
///
/// # Errors
///
/// There are two possible error cases. Both return the unit type [`core::fmt::Error`].
///
/// - In case the formatting exceeds the string's capacity. This error does not exist in
/// the standard library as the string would just grow.
/// - If a formatting trait implementation returns an error. The standard library panics
/// in this case.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), core::fmt::Error> {
/// use heapless::{format2, ArrayString};
///
/// // Notice semicolon instead of comma!
/// format2!(4; "test")?;
/// format2!(15; "hello {}", "world!")?;
/// format2!(20; "x = {}, y = {y}", 10, y = 30)?;
/// let (x, y) = (1, 2);
/// format2!(12; "{x} + {y} = 3")?;
///
/// let implicit: ArrayString<10> = format2!("speed = {}", 7)?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! format2 {
    // Without semicolon as separator to disambiguate between arms, Rust just
    // chooses the first so that the format string would land in $max.
    ($max:expr; $($arg:tt)*) => {{
        let res = $crate::_export::format2::<$max>(core::format_args!($($arg)*));
        res
    }};
    ($($arg:tt)*) => {{
        let res = $crate::_export::format2(core::format_args!($($arg)*));
        res
    }};
}

macro_rules! impl_try_from_num {
    ($num:ty, $size:expr) => {
        impl<const N: usize> core::convert::TryFrom<$num> for ArrayString<N> {
            type Error = ();
            fn try_from(s: $num) -> Result<Self, Self::Error> {
                let mut new = ArrayString::new();
                write!(&mut new, "{}", s).map_err(|_| ())?;
                Ok(new)
            }
        }
    };
}

impl_try_from_num!(i8, 4);
impl_try_from_num!(i16, 6);
impl_try_from_num!(i32, 11);
impl_try_from_num!(i64, 20);

impl_try_from_num!(u8, 3);
impl_try_from_num!(u16, 5);
impl_try_from_num!(u32, 10);
impl_try_from_num!(u64, 20);

#[cfg(test)]
mod tests {
    use crate::{ArrayString, Vec};

    #[test]
    fn static_new() {
        static mut _S: ArrayString<8> = ArrayString::new();
    }

    #[test]
    fn clone() {
        let s1: ArrayString<20> = ArrayString::try_from("abcd").unwrap();
        let mut s2 = s1.clone();
        s2.push_str(" efgh").unwrap();

        assert_eq!(s1, "abcd");
        assert_eq!(s2, "abcd efgh");
    }

    #[test]
    fn cmp() {
        let s1: ArrayString<4> = ArrayString::try_from("abcd").unwrap();
        let s2: ArrayString<4> = ArrayString::try_from("zzzz").unwrap();

        assert!(s1 < s2);
    }

    #[test]
    fn cmp_heterogenous_size() {
        let s1: ArrayString<4> = ArrayString::try_from("abcd").unwrap();
        let s2: ArrayString<8> = ArrayString::try_from("zzzz").unwrap();

        assert!(s1 < s2);
    }

    #[test]
    fn debug() {
        use core::fmt::Write;

        let s: ArrayString<8> = ArrayString::try_from("abcd").unwrap();
        let mut std_s = std::string::String::new();
        write!(std_s, "{:?}", s).unwrap();
        assert_eq!("\"abcd\"", std_s);
    }

    #[test]
    fn display() {
        use core::fmt::Write;

        let s: ArrayString<8> = ArrayString::try_from("abcd").unwrap();
        let mut std_s = std::string::String::new();
        write!(std_s, "{}", s).unwrap();
        assert_eq!("abcd", std_s);
    }

    #[test]
    fn empty() {
        let s: ArrayString<4> = ArrayString::new();
        assert!(s.capacity() == 4);
        assert_eq!(s, "");
        assert_eq!(s.len(), 0);
        assert_ne!(s.len(), 4);
    }

    #[test]
    fn try_from() {
        let s: ArrayString<4> = ArrayString::try_from("123").unwrap();
        assert!(s.len() == 3);
        assert_eq!(s, "123");

        let _: () = ArrayString::<2>::try_from("123").unwrap_err();
    }

    #[test]
    fn from_str() {
        use core::str::FromStr;

        let s: ArrayString<4> = ArrayString::<4>::from_str("123").unwrap();
        assert!(s.len() == 3);
        assert_eq!(s, "123");

        let _: () = ArrayString::<2>::from_str("123").unwrap_err();
    }

    #[test]
    fn from_iter() {
        let mut v: Vec<char, 5> = Vec::new();
        v.push('h').unwrap();
        v.push('e').unwrap();
        v.push('l').unwrap();
        v.push('l').unwrap();
        v.push('o').unwrap();
        let string1: ArrayString<5> = v.iter().collect(); //&char
        let string2: ArrayString<5> = "hello".chars().collect(); //char
        assert_eq!(string1, "hello");
        assert_eq!(string2, "hello");
    }

    #[test]
    #[should_panic]
    fn from_panic() {
        let _: ArrayString<4> = ArrayString::try_from("12345").unwrap();
    }

    #[test]
    fn try_from_num() {
        let v: ArrayString<20> = ArrayString::try_from(18446744073709551615_u64).unwrap();
        assert_eq!(v, "18446744073709551615");

        let _: () = ArrayString::<2>::try_from(18446744073709551615_u64).unwrap_err();
    }

    #[test]
    fn into_bytes() {
        let s: ArrayString<4> = ArrayString::try_from("ab").unwrap();
        let b: Vec<u8, 4> = s.into_bytes();
        assert_eq!(b.len(), 2);
        assert_eq!(&[b'a', b'b'], &b[..]);
    }

    #[test]
    fn as_str() {
        let s: ArrayString<4> = ArrayString::try_from("ab").unwrap();

        assert_eq!(s.as_str(), "ab");
        // should be moved to fail test
        //    let _s = s.as_str();
        // s.push('c'); // <- cannot borrow `s` as mutable because it is also borrowed as immutable
    }

    #[test]
    fn as_mut_str() {
        let mut s: ArrayString<4> = ArrayString::try_from("ab").unwrap();
        let s = s.as_mut_str();
        s.make_ascii_uppercase();
        assert_eq!(s, "AB");
    }

    #[test]
    fn push_str() {
        let mut s: ArrayString<8> = ArrayString::try_from("foo").unwrap();
        assert!(s.push_str("bar").is_ok());
        assert_eq!("foobar", s);
        assert_eq!(s, "foobar");
        assert!(s.push_str("tender").is_err());
        assert_eq!("foobar", s);
        assert_eq!(s, "foobar");
    }

    #[test]
    fn push() {
        let mut s: ArrayString<6> = ArrayString::try_from("abc").unwrap();
        assert!(s.push('1').is_ok());
        assert!(s.push('2').is_ok());
        assert!(s.push('3').is_ok());
        assert!(s.push('4').is_err());
        assert!("abc123" == s.as_str());
    }

    #[test]
    fn as_bytes() {
        let s: ArrayString<8> = ArrayString::try_from("hello").unwrap();
        assert_eq!(&[104, 101, 108, 108, 111], s.as_bytes());
    }

    #[test]
    fn truncate() {
        let mut s: ArrayString<8> = ArrayString::try_from("hello").unwrap();
        s.truncate(6);
        assert_eq!(s.len(), 5);
        s.truncate(2);
        assert_eq!(s.len(), 2);
        assert_eq!("he", s);
        assert_eq!(s, "he");
    }

    #[test]
    fn pop() {
        let mut s: ArrayString<8> = ArrayString::try_from("foo").unwrap();
        println!("S LENGTH: {}", s.len());
        assert_eq!(s.pop(), Some('o'));
        assert_eq!(s.pop(), Some('o'));
        assert_eq!(s.pop(), Some('f'));
        assert_eq!(s.pop(), None);
    }

    #[test]
    fn pop_uenc() {
        let mut s: ArrayString<8> = ArrayString::try_from("é").unwrap();
        assert_eq!(s.len(), 3);
        println!("S: {s}");
        match s.pop() {
            Some(c) => {
                assert_eq!(s.len(), 1);
                assert_eq!(c, '\u{0301}'); // accute accent of e
            }
            None => panic!(),
        };
    }

    #[test]
    fn is_empty() {
        let mut v: ArrayString<8> = ArrayString::new();
        assert!(v.is_empty());
        let _ = v.push('a');
        assert!(!v.is_empty());
    }

    #[test]
    fn clear() {
        let mut s: ArrayString<8> = ArrayString::try_from("foo").unwrap();
        s.clear();
        assert!(s.is_empty());
        assert_eq!(0, s.len());
        assert_eq!(8, s.capacity());
    }

    /*

    #[test]
    fn remove() {
        let mut s: ArrayString<8> = ArrayString::try_from("foo").unwrap();
        assert_eq!(s.remove(0), 'f');
        assert_eq!(s.as_str(), "oo");
    }

    #[test]
    fn remove_uenc() {
        let mut s: ArrayString<8> = ArrayString::try_from("ĝėēƶ").unwrap();
        assert_eq!(s.remove(2), 'ė');
        assert_eq!(s.remove(2), 'ē');
        assert_eq!(s.remove(2), 'ƶ');
        assert_eq!(s.as_str(), "ĝ");
    }

    #[test]
    fn remove_uenc_combo_characters() {
        let mut s: ArrayString<8> = ArrayString::try_from("héy").unwrap();
        assert_eq!(s.remove(2), '\u{0301}');
        assert_eq!(s.as_str(), "hey");
    }
    */

    #[test]
    fn format() {
        let number = 5;
        let float = 3.12;
        let formatted = format2!(15; "{:0>3} plus {float}", number).unwrap();
        assert_eq!(formatted, "005 plus 3.12")
    }
    #[test]
    fn format_inferred_capacity() {
        let number = 5;
        let float = 3.12;
        let formatted: ArrayString<15> = format2!("{:0>3} plus {float}", number).unwrap();
        assert_eq!(formatted, "005 plus 3.12")
    }

    #[test]
    fn format_overflow() {
        let i = 1234567;
        let formatted = format2!(4; "13{}", i);
        assert_eq!(formatted, Err(core::fmt::Error))
    }

    #[test]
    fn format_plain_string_overflow() {
        let formatted = format2!(2; "123");
        assert_eq!(formatted, Err(core::fmt::Error))
    }
}
