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
             py::arg("n_agents")  = 1,
             py::arg("spawn_x")   = 320.f,
             py::arg("spawn_y")   = 170.f,
             py::arg("render")    = true,
             py::arg("scene_idx") = 5,
             "Initialise SDL + Vulkan + physics and spawn n_agents characters. "
             "Agents are spaced AGENT_SPACING px apart horizontally from spawn_x.")
        .def("step",
             [](PyEnv& self,
                const std::vector<std::vector<float>>& all_angles,
                const std::vector<float>& move_dirs,
                const std::vector<bool>&  jumps,
                const std::vector<float>& aim_xs,
                const std::vector<float>& aim_ys,
                bool sim_inputs, bool render) {
                 py::gil_scoped_release release;
                 auto [obs, rewards, dones] = self.step(
                     all_angles, move_dirs, jumps, aim_xs, aim_ys, sim_inputs, render);
                 py::gil_scoped_acquire acquire;
                 return py::make_tuple(obs, rewards, dones);
             },
             py::arg("all_angles"),
             py::arg("move_dirs")  = std::vector<float>{},
             py::arg("jumps")      = std::vector<bool>{},
             py::arg("aim_xs")     = std::vector<float>{},
             py::arg("aim_ys")     = std::vector<float>{},
             py::arg("sim_inputs") = false,
             py::arg("render")     = true,
             "Batched step. Returns (obs[n][OBS_SIZE], rewards[n], dones[n]). "
             "Empty move_dirs/jumps/aim_xs/aim_ys default to zeros/false.")
        .def("reset_agent",
             &PyEnv::reset_agent,
             py::arg("i"),
             py::arg("spawn_x") = -1.f,
             py::arg("spawn_y") = -1.f,
             "Teleport agent i to its spawn position and zero velocities. "
             "Pass spawn_x/spawn_y >= 0 to override the stored spawn position.")
        .def("reset_all",
             &PyEnv::reset_all,
             py::arg("spawn_x") = -1.f,
             py::arg("spawn_y") = -1.f,
             "Reset all agents.")
        .def("get_agent_obs",
             &PyEnv::get_agent_obs,
             py::arg("i"),
             "Return the current observation vector for agent i.")
        .def("get_obs_all",
             &PyEnv::get_obs_all,
             "Return observation vectors for all agents as a list of lists.")
        .def("close",
             &PyEnv::close,
             "Shut down SDL and Vulkan.")
        .def_property_readonly("n_agents",       &PyEnv::n_agents)
        .def_property_readonly("obs_size",       [](const PyEnv&) { return PyEnv::OBS_SIZE; })
        .def_property_readonly("frame_obs_size", [](const PyEnv&) { return PyEnv::FRAME_OBS_SIZE; })
        .def_property_readonly("history_len",    [](const PyEnv&) { return (int)STATE_HISTORY_LEN; })
        .def_property_readonly("joint_count",    [](const PyEnv&) { return (int)JOINT_COUNT; })
        .def_property_readonly("limb_count",     [](const PyEnv&) { return (int)Limb::Count; });
}
