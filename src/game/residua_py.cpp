#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_env.h"

namespace py = pybind11;

PYBIND11_MODULE(residua_py, m) {
    m.doc() = "Residua physics simulation — Python bindings";

    py::class_<PyEnv>(m, "PyEnv")
        .def(py::init<>())
        .def("init",
             &PyEnv::init,
             py::arg("spawn_x")   = 320.f,
             py::arg("spawn_y")   = 170.f,
             py::arg("render")    = true,
             py::arg("scene_idx") = 5,
             "Initialise SDL + Vulkan + physics and spawn the character. "
             "render=False hides the window for fast headless training.")
        .def("step",
             [](PyEnv& self,
                const std::vector<float>& torques,
                float aim_x, float aim_y,
                bool render) {
                 py::gil_scoped_release release;
                 auto [obs, reward, done] = self.step(torques, { aim_x, aim_y }, render);
                 py::gil_scoped_acquire acquire;
                 return py::make_tuple(obs, reward, done);
             },
             py::arg("torques"),
             py::arg("aim_x")  = 400.f,
             py::arg("aim_y")  = 170.f,
             py::arg("render") = true,
             "Step the simulation. Returns (obs[28], reward, done).")
        .def("reset",
             &PyEnv::reset,
             py::arg("spawn_x") = 320.f,
             py::arg("spawn_y") = 170.f,
             "Despawn and respawn the character with zero velocities.")
        .def("close",
             &PyEnv::close,
             "Shut down SDL and Vulkan.")
        .def_property_readonly("obs_size",
             [](const PyEnv&) { return PyEnv::OBS_SIZE; })
        .def_property_readonly("limb_count",
             [](const PyEnv&) { return (int)Limb::Count; });
}
