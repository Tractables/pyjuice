// The Triton fast launcher's (`pyjuice.utils.kernel_launcher.FastJITFunction3x`) per-launch work in C++.
//
// `bind(dev, values, constexpr, meta)` builds the values a launch runs with and the signature its compiled kernel is
// cached under exactly as the launcher's generated Python binding does --
//     a constexpr                     its value, in both;
//     a tensor                        (dtype, data_ptr & 15) in the signature, and its address (an int) in the values if
//                                     it is on the current CUDA device `dev`, else the tensor itself (a CPU tensor, or
//                                     one on another GPU, whose memory Triton's launcher checks);
//     any other value                 itself, in both;
// the signature (dev, *those, meta) -- in one call instead of a few attribute reads per tensor. Its domain is narrow:
// a tensor that is not exactly a `Tensor` or a `Parameter`, not strided, or not on the CPU or a CUDA device, or any
// value under a TorchFunctionMode (which the Python binding's attribute reads go through), and it returns None -- the
// Python binding binds that call.
//
// `launcher(state, grid)` is what `kernel[grid]` returns: a callable launching as the launcher's Python path
// (`FastJITFunction3x._launch`) does, for the calls it takes -- a call pattern (how many arguments by position, the
// keywords' names in order) whose place of every parameter among the call's arguments the Python path registered (a
// spec), values in `bind`'s domain, a compiled kernel cached under their signature. It reads the current device and
// stream with the functions the Python path reads them with, binds as `bind` does, looks the kernel up and calls its
// `run` with the arguments the Python path passes it. Any other call it hands to the Python path, which binds,
// compiles and caches as before; one raising an `Exception` on the way, to Triton's own launch, as the Python path
// does (other exceptions propagate).

#include <Python.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

namespace {

// An owned reference, released unless handed over
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* p_ = nullptr) : p(p_) {}
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { Py_XDECREF(p); }
    PyObject* release() {
        PyObject* q = p;
        p = nullptr;
        return q;
    }
};

// A tensor's signature entry (dtype, data_ptr & 15), one tuple per (dtype, alignment) for the module's life: equal
// to (and hashing as) the Python binding's fresh one
PyObject* aligned[static_cast<int>(at::ScalarType::NumOptions)][16] = {};

PyObject* aligned_entry(at::ScalarType st, unsigned low) {
    PyObject*& e = aligned[static_cast<int>(st)][low];
    if (e == nullptr) {
        PyObject* dtype = reinterpret_cast<PyObject*>(torch::getTHPDtype(st));
        Ref bits(PyLong_FromUnsignedLong(low));
        if (bits.p == nullptr) return nullptr;
        e = PyTuple_Pack(2, dtype, bits.p);
        if (e == nullptr) return nullptr;
    }
    Py_INCREF(e);
    return e;
}

enum Bound { BOUND, DECLINED, FAILED };

// The binding of the `n` values (borrowed references): `run[i]` a new reference to the value to run with, `*sig_out`
// to the signature (dev, ..., meta) -- BOUND; DECLINED, nothing held, for a value outside the domain; FAILED, nothing
// held, with an error set
Bound bind_into(PyObject* dev_obj, long dev, PyObject* const* values, Py_ssize_t n, const char* constexpr_,
                PyObject* meta, PyObject** run, PyObject** sig_out) {
    if (at::impl::torch_function_mode_enabled()) return DECLINED;   // (the Python binding's reads go through it)
    Ref sig(PyTuple_New(n + 2));
    if (sig.p == nullptr) return FAILED;
    Py_INCREF(dev_obj);
    PyTuple_SET_ITEM(sig.p, 0, dev_obj);
    Bound out = BOUND;
    Py_ssize_t i = 0;
    for (; i < n; ++i) {
        PyObject* v = values[i];
        if (constexpr_[i] != 0 || !THPVariable_Check(v)) {          // by value, in both
            Py_INCREF(v);
            PyTuple_SET_ITEM(sig.p, i + 1, v);
            Py_INCREF(v);
            run[i] = v;
            continue;
        }
        if (!THPVariable_CheckExact(v)) {                            // (a subclass: the Python binding's)
            out = DECLINED;
            break;
        }
        PyObject* entry = nullptr;
        PyObject* value = nullptr;
        try {
            const at::Tensor& t = THPVariable_Unpack(v);
            if (t.layout() != at::kStrided || !(t.is_cpu() || t.is_cuda())) {
                out = DECLINED;
                break;
            }
            const uintptr_t p = reinterpret_cast<uintptr_t>(t.data_ptr());
            entry = aligned_entry(t.scalar_type(), static_cast<unsigned>(p & 15));
            if (entry == nullptr) {
                out = FAILED;
                break;
            }
            if (t.get_device() == dev) {                              // (a CPU tensor's is -1)
                value = PyLong_FromUnsignedLongLong(p);
                if (value == nullptr) {
                    Py_DECREF(entry);
                    out = FAILED;
                    break;
                }
            } else {
                Py_INCREF(v);
                value = v;
            }
        } catch (const std::exception& e) {                           // (e.g. a tensor without storage)
            Py_XDECREF(entry);
            PyErr_SetString(PyExc_RuntimeError, e.what());
            out = FAILED;
            break;
        }
        PyTuple_SET_ITEM(sig.p, i + 1, entry);
        run[i] = value;
    }
    if (out != BOUND) {
        for (Py_ssize_t j = 0; j < i; ++j) Py_DECREF(run[j]);
        return out;                                                   // (the signature's unset items are NULL)
    }
    Py_INCREF(meta);
    PyTuple_SET_ITEM(sig.p, n + 1, meta);
    *sig_out = sig.release();
    return BOUND;
}

// bind(dev, values, constexpr, meta) -> (values to run with, signature), or None; `values` the parameters' in
// declared order, `constexpr` a byte per parameter (non-zero: a constexpr), `meta` the launch's meta-parameters
PyObject* bind(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 4 || !PyLong_Check(args[0]) || !PyTuple_Check(args[1]) || !PyBytes_Check(args[2])) {
        PyErr_SetString(PyExc_TypeError, "bind(dev: int, values: tuple, constexpr: bytes, meta)");
        return nullptr;
    }
    PyObject *dev_obj = args[0], *values = args[1], *meta = args[3];
    const Py_ssize_t n = PyTuple_GET_SIZE(values);
    if (PyBytes_GET_SIZE(args[2]) != n) {
        PyErr_SetString(PyExc_ValueError, "bind: a constexpr byte per value");
        return nullptr;
    }
    const long dev = PyLong_AsLong(dev_obj);
    if (dev == -1 && PyErr_Occurred()) return nullptr;
    std::vector<PyObject*> run(static_cast<size_t>(n));
    PyObject* sig = nullptr;
    switch (bind_into(dev_obj, dev, PySequence_Fast_ITEMS(values), n, PyBytes_AS_STRING(args[2]), meta, run.data(),
                      &sig)) {
        case DECLINED: Py_RETURN_NONE;
        case FAILED: return nullptr;
        case BOUND: break;
    }
    Ref sigr(sig), runt(PyTuple_New(n));
    if (runt.p == nullptr) {
        for (PyObject* v : run) Py_DECREF(v);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < n; ++i) PyTuple_SET_ITEM(runt.p, i, run[static_cast<size_t>(i)]);
    PyObject* out = PyTuple_New(2);
    if (out == nullptr) return nullptr;
    PyTuple_SET_ITEM(out, 0, runt.release());
    PyTuple_SET_ITEM(out, 1, sigr.release());
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// The launcher
// ---------------------------------------------------------------------------------------------------------------

// The launcher's state, a tuple `FastJITFunction3x._native_state` builds: the call patterns' specs, (count of
// positional arguments, keyword names) -> (every parameter's place among the call's arguments, 2 bytes each, little
// endian: 1 + the place, 0 for its default; the parameters' defaults; the meta-parameters given, ((name, place), ...)
// sorted by name); the compiled kernels by signature; the constexpr bytes; the Python path (grid, args, kwargs); the
// standard launch (grid, args, kwargs); `triton.knobs`; its `HookChain` type; the current device's and stream's
// getters
enum { S_SPECS, S_CACHE, S_CONSTEXPR, S_PYTHON, S_STANDARD, S_KNOBS, S_HOOK_CHAIN, S_DEVICE, S_STREAM, S_SIZE };

// The most parameters a launch the launcher takes has (more: the Python path's)
constexpr Py_ssize_t MOST_PARAMS = 240;

struct Launcher {
    PyObject_HEAD
    vectorcallfunc call;
    PyObject* state;
    PyObject* grid;
};

PyObject *s_runtime, *s_enter, *s_exit, *s_calls, *s_run, *s_function, *s_packed, *one, *no_keywords;

// The call (its `nargs` positional arguments, then the values of the keywords `kwnames`) handed to `fn(grid, args,
// kwargs)`
PyObject* hand_over(PyObject* fn, PyObject* grid, PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames) {
    Ref a(PyTuple_New(nargs));
    if (a.p == nullptr) return nullptr;
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        Py_INCREF(args[i]);
        PyTuple_SET_ITEM(a.p, i, args[i]);
    }
    Ref k(PyDict_New());
    if (k.p == nullptr) return nullptr;
    const Py_ssize_t nkw = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t j = 0; j < nkw; ++j)                             // (in the keywords' order)
        if (PyDict_SetItem(k.p, PyTuple_GET_ITEM(kwnames, j), args[nargs + j]) < 0) return nullptr;
    return PyObject_CallFunctionObjArgs(fn, grid, a.p, k.p, nullptr);
}

// A launch hook (a new reference) as the launch passes it: None for a HookChain with no hook in it -- which Triton's
// launcher would call to do nothing --, else the hook
PyObject* hook_argument(PyObject* runtime, PyObject* name, PyObject* chain_type) {
    PyObject* hook = PyObject_GetAttr(runtime, name);
    if (hook == nullptr || reinterpret_cast<PyObject*>(Py_TYPE(hook)) != chain_type) return hook;
    PyObject* calls = PyObject_GetAttr(hook, s_calls);
    if (calls == nullptr) {
        Py_DECREF(hook);
        return nullptr;
    }
    const bool empty = PyList_CheckExact(calls) && PyList_GET_SIZE(calls) == 0;
    Py_DECREF(calls);
    if (!empty) return hook;
    Py_DECREF(hook);
    Py_RETURN_NONE;
}

// The launch, if the launcher takes the call: a new reference (None) once launched; nullptr with an error set; or
// nullptr without one, the call declined
PyObject* fast_launch(Launcher* L, PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames) {
    PyObject* state = L->state;
    Ref count(PyLong_FromSsize_t(nargs));
    if (count.p == nullptr) return nullptr;
    Ref key(PyTuple_Pack(2, count.p, kwnames == nullptr ? no_keywords : kwnames));
    if (key.p == nullptr) return nullptr;
    PyObject* spec = PyDict_GetItemWithError(PyTuple_GET_ITEM(state, S_SPECS), key.p);
    if (spec == nullptr) return nullptr;                             // (an unseen pattern, or an error)
    if (!PyTuple_CheckExact(spec) || PyTuple_GET_SIZE(spec) != 3) return nullptr;
    PyObject* places = PyTuple_GET_ITEM(spec, 0);
    PyObject* defaults = PyTuple_GET_ITEM(spec, 1);
    PyObject* meta_places = PyTuple_GET_ITEM(spec, 2);
    PyObject* constexpr_ = PyTuple_GET_ITEM(state, S_CONSTEXPR);
    if (!PyBytes_CheckExact(places) || !PyTuple_CheckExact(defaults) || !PyTuple_CheckExact(meta_places)) return nullptr;
    const Py_ssize_t n = PyTuple_GET_SIZE(defaults), given = nargs + (kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames));
    if (n > MOST_PARAMS || PyBytes_GET_SIZE(places) != 2 * n || PyBytes_GET_SIZE(constexpr_) != n) return nullptr;
    const unsigned char* at = reinterpret_cast<const unsigned char*>(PyBytes_AS_STRING(places));
    PyObject* values[MOST_PARAMS];                                   // (borrowed)
    for (Py_ssize_t i = 0; i < n; ++i) {
        const Py_ssize_t place = at[2 * i] | (static_cast<Py_ssize_t>(at[2 * i + 1]) << 8);
        if (place > given) return nullptr;                           // (a spec not of this pattern)
        values[i] = place == 0 ? PyTuple_GET_ITEM(defaults, i) : args[place - 1];
    }
    const Py_ssize_t nm = PyTuple_GET_SIZE(meta_places);
    Ref meta(nm == 0 ? (Py_INCREF(no_keywords), no_keywords) : PyTuple_New(nm));
    if (meta.p == nullptr) return nullptr;
    for (Py_ssize_t j = 0; j < nm; ++j) {
        PyObject* e = PyTuple_GET_ITEM(meta_places, j);
        if (!PyTuple_CheckExact(e) || PyTuple_GET_SIZE(e) != 2) return nullptr;
        const Py_ssize_t place = PyLong_AsSsize_t(PyTuple_GET_ITEM(e, 1));
        if (place == -1 && PyErr_Occurred()) return nullptr;
        if (place < 0 || place >= given) return nullptr;
        PyObject* item = PyTuple_Pack(2, PyTuple_GET_ITEM(e, 0), args[place]);
        if (item == nullptr) return nullptr;
        PyTuple_SET_ITEM(meta.p, j, item);
    }
    Ref dev_obj(PyObject_CallNoArgs(PyTuple_GET_ITEM(state, S_DEVICE)));
    if (dev_obj.p == nullptr) return nullptr;
    const long dev = PyLong_AsLong(dev_obj.p);
    if (dev == -1 && PyErr_Occurred()) return nullptr;
    PyObject* call[9 + MOST_PARAMS];                                 // (the run's arguments; owned past the first 9)
    PyObject** run = call + 9;
    PyObject* sig = nullptr;
    switch (bind_into(dev_obj.p, dev, values, n, PyBytes_AS_STRING(constexpr_), meta.p, run, &sig)) {
        case DECLINED: return nullptr;
        case FAILED: return nullptr;
        case BOUND: break;
    }
    struct Held {                                                    // (the bound values, released on the way out)
        PyObject** v;
        Py_ssize_t n;
        ~Held() {
            for (Py_ssize_t i = 0; i < n; ++i) Py_DECREF(v[i]);
        }
    } held{run, n};
    Ref sigr(sig);
    PyObject* kernel = PyDict_GetItemWithError(PyTuple_GET_ITEM(state, S_CACHE), sig);
    if (kernel == nullptr) return nullptr;                           // (not compiled yet, or an error)
    Ref kernelr((Py_INCREF(kernel), kernel));
    PyObject* grid = L->grid;
    const Py_ssize_t ng = PyTuple_GET_SIZE(grid);
    if (ng == 0) {
        PyErr_SetString(PyExc_IndexError, "tuple index out of range");   // (as `grid[0]` raises)
        return nullptr;
    }
    Ref stream(PyObject_CallOneArg(PyTuple_GET_ITEM(state, S_STREAM), dev_obj.p));
    if (stream.p == nullptr) return nullptr;
    Ref runtime(PyObject_GetAttr(PyTuple_GET_ITEM(state, S_KNOBS), s_runtime));
    if (runtime.p == nullptr) return nullptr;
    PyObject* chain = PyTuple_GET_ITEM(state, S_HOOK_CHAIN);
    Ref enter(hook_argument(runtime.p, s_enter, chain));
    if (enter.p == nullptr) return nullptr;
    Ref exit_(hook_argument(runtime.p, s_exit, chain));
    if (exit_.p == nullptr) return nullptr;
    Ref fn(PyObject_GetAttr(kernel, s_run)), function(PyObject_GetAttr(kernel, s_function)),
        packed(PyObject_GetAttr(kernel, s_packed));
    if (fn.p == nullptr || function.p == nullptr || packed.p == nullptr) return nullptr;
    call[0] = PyTuple_GET_ITEM(grid, 0);
    call[1] = ng > 1 ? PyTuple_GET_ITEM(grid, 1) : one;
    call[2] = ng > 2 ? PyTuple_GET_ITEM(grid, 2) : one;
    call[3] = stream.p;
    call[4] = function.p;
    call[5] = packed.p;
    call[6] = Py_None;                                               // (launch metadata: for hooks alone)
    call[7] = enter.p;
    call[8] = exit_.p;
    PyObject* r = PyObject_Vectorcall(fn.p, call, static_cast<size_t>(9 + n), nullptr);
    if (r == nullptr) return nullptr;
    Py_DECREF(r);
    Py_RETURN_NONE;
}

PyObject* launcher_call(PyObject* self, PyObject* const* args, size_t nargsf, PyObject* kwnames) {
    Launcher* L = reinterpret_cast<Launcher*>(self);
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyObject* r = nullptr;
    try {
        r = fast_launch(L, args, nargs, kwnames);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    if (r != nullptr) return r;
    if (!PyErr_Occurred())                                           // declined: the Python path's
        return hand_over(PyTuple_GET_ITEM(L->state, S_PYTHON), L->grid, args, nargs, kwnames);
    if (!PyErr_ExceptionMatches(PyExc_Exception)) return nullptr;   // (KeyboardInterrupt, SystemExit: raised)
    PyErr_Clear();                                                   // an error on the way: the standard launch
    return hand_over(PyTuple_GET_ITEM(L->state, S_STANDARD), L->grid, args, nargs, kwnames);
}

void launcher_dealloc(PyObject* self) {
    Launcher* L = reinterpret_cast<Launcher*>(self);
    Py_XDECREF(L->state);
    Py_XDECREF(L->grid);
    PyObject_Free(self);
}

PyTypeObject LauncherType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// launcher(state, grid) -> a launch of `grid`, called as the kernel is
PyObject* launcher(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2 || !PyTuple_CheckExact(args[0]) || PyTuple_GET_SIZE(args[0]) != S_SIZE ||
        !PyTuple_CheckExact(args[1]) || !PyDict_CheckExact(PyTuple_GET_ITEM(args[0], S_SPECS)) ||
        !PyDict_CheckExact(PyTuple_GET_ITEM(args[0], S_CACHE)) ||
        !PyBytes_CheckExact(PyTuple_GET_ITEM(args[0], S_CONSTEXPR))) {
        PyErr_SetString(PyExc_TypeError, "launcher(state: tuple, grid: tuple)");
        return nullptr;
    }
    Launcher* L = PyObject_New(Launcher, &LauncherType);
    if (L == nullptr) return nullptr;
    L->call = launcher_call;
    Py_INCREF(args[0]);
    L->state = args[0];
    Py_INCREF(args[1]);
    L->grid = args[1];
    return reinterpret_cast<PyObject*>(L);
}

PyMethodDef methods[] = {
    {"bind", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(bind)), METH_FASTCALL,
     "bind(dev, values, constexpr, meta) -> (values to run with, signature), or None (see launch_bind.cpp)"},
    {"launcher", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(launcher)), METH_FASTCALL,
     "launcher(state, grid) -> a callable launching the kernel over grid (see launch_bind.cpp)"},
    {nullptr, nullptr, 0, nullptr}};

#define PJ_STR_(name) #name
#define PJ_STR(name) PJ_STR_(name)
#define PJ_INIT_(name) PyInit_##name
#define PJ_INIT(name) PJ_INIT_(name)

PyModuleDef module = {PyModuleDef_HEAD_INIT, PJ_STR(TORCH_EXTENSION_NAME),
                      "The fast launcher's argument binding and launch.", -1, methods};

}  // namespace

PyMODINIT_FUNC PJ_INIT(TORCH_EXTENSION_NAME)(void) {
    LauncherType.tp_name = PJ_STR(TORCH_EXTENSION_NAME) ".Launcher";
    LauncherType.tp_basicsize = sizeof(Launcher);
    LauncherType.tp_dealloc = launcher_dealloc;
    LauncherType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL;
    LauncherType.tp_vectorcall_offset = offsetof(Launcher, call);
    LauncherType.tp_call = PyVectorcall_Call;
    LauncherType.tp_doc = "A launch of a grid, called as the kernel is (see launch_bind.cpp)";
    if (PyType_Ready(&LauncherType) < 0) return nullptr;
    s_runtime = PyUnicode_InternFromString("runtime");
    s_enter = PyUnicode_InternFromString("launch_enter_hook");
    s_exit = PyUnicode_InternFromString("launch_exit_hook");
    s_calls = PyUnicode_InternFromString("calls");
    s_run = PyUnicode_InternFromString("run");
    s_function = PyUnicode_InternFromString("function");
    s_packed = PyUnicode_InternFromString("packed_metadata");
    one = PyLong_FromLong(1);
    no_keywords = PyTuple_New(0);
    if (!s_runtime || !s_enter || !s_exit || !s_calls || !s_run || !s_function || !s_packed || !one || !no_keywords)
        return nullptr;
    return PyModule_Create(&module);
}
