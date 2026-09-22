from setuptools import setup

package_name = "rsv_hw"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Masimba249",
    maintainer_email="Masimba249@gmail.com",
    description="Hardware interface for rare-event safety validation with a learned digital twin.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "recorder = rsv_hw.recorder_node:main",
            "scenario = rsv_hw.scenario_node:main",
        ],
    },
)
