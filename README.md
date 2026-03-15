# blank-thoughts
eg : 


cd "C:\Users\james\OneDrive\Documents\Coding Projects\blank-thoughts"
Remove-Item -Recurse -Force build
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
.\build\Release\SolarSystem.exe