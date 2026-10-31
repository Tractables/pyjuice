// The per-argument binding of the Triton fast launcher (`pyjuice.utils.kernel_launcher.FastJITFunction3x`) in C++:
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
// Python binding binds that call. Nothing here launches or touches Triton.

#include <Python.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/python_variable.h>

#include <exception>

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
    const char* constexpr_ = PyBytes_AS_STRING(args[2]);
    if (at::impl::torch_function_mode_enabled()) Py_RETURN_NONE;   // (the Python binding's reads go through it)

    Ref run(PyTuple_New(n)), sig(PyTuple_New(n + 2));
    if (run.p == nullptr || sig.p == nullptr) return nullptr;
    Py_INCREF(dev_obj);
    PyTuple_SET_ITEM(sig.p, 0, dev_obj);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* v = PyTuple_GET_ITEM(values, i);
        if (constexpr_[i] != 0 || !THPVariable_Check(v)) {          // by value, in both
            Py_INCREF(v);
            PyTuple_SET_ITEM(sig.p, i + 1, v);
            Py_INCREF(v);
            PyTuple_SET_ITEM(run.p, i, v);
            continue;
        }
        if (!THPVariable_CheckExact(v)) Py_RETURN_NONE;              // (a subclass: the Python binding's)
        PyObject* entry = nullptr;
        PyObject* value = nullptr;
        try {
            const at::Tensor& t = THPVariable_Unpack(v);
            if (t.layout() != at::kStrided || !(t.is_cpu() || t.is_cuda())) Py_RETURN_NONE;
            const uintptr_t p = reinterpret_cast<uintptr_t>(t.data_ptr());
            entry = aligned_entry(t.scalar_type(), static_cast<unsigned>(p & 15));
            if (entry == nullptr) return nullptr;
            if (t.get_device() == dev) {                              // (a CPU tensor's is -1)
                value = PyLong_FromUnsignedLongLong(p);
                if (value == nullptr) {
                    Py_DECREF(entry);
                    return nullptr;
                }
            } else {
                Py_INCREF(v);
                value = v;
            }
        } catch (const std::exception& e) {                           // (e.g. a tensor without storage)
            Py_XDECREF(entry);
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
        }
        PyTuple_SET_ITEM(sig.p, i + 1, entry);
        PyTuple_SET_ITEM(run.p, i, value);
    }
    Py_INCREF(meta);
    PyTuple_SET_ITEM(sig.p, n + 1, meta);
    PyObject* out = PyTuple_New(2);
    if (out == nullptr) return nullptr;
    PyTuple_SET_ITEM(out, 0, run.release());
    PyTuple_SET_ITEM(out, 1, sig.release());
    return out;
}

PyMethodDef methods[] = {
    {"bind", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(bind)), METH_FASTCALL,
     "bind(dev, values, constexpr, meta) -> (values to run with, signature), or None (see launch_bind.cpp)"},
    {nullptr, nullptr, 0, nullptr}};

#define PJ_STR_(name) #name
#define PJ_STR(name) PJ_STR_(name)
#define PJ_INIT_(name) PyInit_##name
#define PJ_INIT(name) PJ_INIT_(name)

PyModuleDef module = {PyModuleDef_HEAD_INIT, PJ_STR(TORCH_EXTENSION_NAME), "The fast launcher's argument binding.", -1,
                      methods};

}  // namespace

PyMODINIT_FUNC PJ_INIT(TORCH_EXTENSION_NAME)(void) {
    return PyModule_Create(&module);
}
