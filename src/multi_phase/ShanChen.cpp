#include "palabos3D.h"
#include "palabos3D.hh"

#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace plb;
using namespace std;

typedef double T;
#define DESCRIPTOR descriptors::ForcedShanChenD3Q19Descriptor

namespace {

const T kSurfaceTension = (T)0.15;
const int kFluid2Tag = 0;
const int kWallTag1 = 1;
const int kSolidTag = 2;
const int kFluid1Tag = 3;
const int kWallTag2 = 4;
const int kMeshTag = 5;
const int kWallTag3 = 6;
const int kWallTag4 = 7;
const int kFieldPrecision = std::numeric_limits<T>::max_digits10;

std::string formatRunId(plint run)
{
    std::ostringstream stream;
    stream << std::setw(3) << std::setfill('0') << run;
    return stream.str();
}

T computeRelativeDifference(T previousValue, T currentValue, plint interval)
{
    if (interval <= 0) {
        return std::numeric_limits<T>::infinity();
    }
    if (previousValue == (T)0) {
        return currentValue == (T)0 ? (T)0 : std::numeric_limits<T>::infinity();
    }
    return std::fabs(previousValue - currentValue) * (T)100
         / std::fabs(previousValue) / (T)interval;
}

Box3D makeOneBasedBox(
    plint x1, plint x2, plint y1, plint y2, plint z1, plint z2)
{
    return Box3D(x1 - 1, x2 - 1, y1 - 1, y2 - 1, z1 - 1, z2 - 1);
}

}  // namespace

void writeDensityGifMidplane(
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid1, const std::string &runId, plint iteration)
{
    const plint imSize = 600;
    const plint nx = latticeFluid1.getNx();
    const plint ny = latticeFluid1.getNy();
    const plint nz = latticeFluid1.getNz();
    Box3D slice(0, nx - 1, 0, ny - 1, nz / 2, nz / 2);

    ImageWriter<T> imageWriter("leeloo.map");
    imageWriter.writeScaledGif(
        createFileName("rho_f1_" + runId + "_", iteration, 8),
        *computeDensity(latticeFluid1, slice), imSize, imSize);
}

void writeDensityGifCenterSlice(
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid1, const std::string &runId, plint iteration)
{
    const plint imSize = 600;
    const plint nx = latticeFluid1.getNx();
    const plint ny = latticeFluid1.getNy();
    const plint nz = latticeFluid1.getNz();
    Box3D slice(0, nx - 1, ny / 2, ny / 2, 0, nz - 1);

    ImageWriter<T> imageWriter("leeloo.map");
    imageWriter.writeScaledGif(
        createFileName("rho_f1_y_" + runId + "_", iteration, 8),
        *computeDensity(latticeFluid1, slice), imSize, imSize);
}

void writeDensityVtk(
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid, const std::string &prefix,
    const std::string &runId, plint iteration)
{
    VtkImageOutput3D<double> vtkOut(createFileName(prefix + runId + "_", iteration, 8), 1.);
    vtkOut.writeData<double>(*computeDensity(latticeFluid), "Density", 1.);
}

T computeAverageXVelocity(
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid, const std::string &label)
{
    const plint nx = latticeFluid.getNx();
    const plint ny = latticeFluid.getNy();
    const plint nz = latticeFluid.getNz();
    Box3D domain(0, nx - 1, 0, ny - 1, 0, nz - 1);

    T meanVelocity = computeAverage(*computeVelocityComponent(latticeFluid, domain, 0));
    pcout << "Average x-velocity for " << label << " [l.u.] = " << meanVelocity << std::endl;
    return meanVelocity;
}

T computeCapillaryNumber(
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid, T viscosity, const std::string &label)
{
    T meanVelocity = computeAverageXVelocity(latticeFluid, label);
    return viscosity * meanVelocity / kSurfaceTension;
}

void readGeometry(
    const std::string &geometryFileName, const std::string &outputDir,
    MultiScalarField3D<int> &geometry, bool printGeometryVtk, bool printGeometryStl)
{
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();
    const plint nz = geometry.getNz();

    plb_ifstream geometryFile(geometryFileName.c_str());
    if (!geometryFile.is_open()) {
        pcout << "Error: could not open geometry file " << geometryFileName << std::endl;
        exit(EXIT_FAILURE);
    }

    Box3D sliceBox(0, 0, 0, ny - 1, 0, nz - 1);
    std::unique_ptr<MultiScalarField3D<int> > slice =
        generateMultiScalarField<int>(geometry, sliceBox);

    for (plint iX = 0; iX < nx - 1; ++iX) {
        geometryFile >> *slice;
        copy(*slice, slice->getBoundingBox(), geometry, Box3D(iX, iX, 0, ny - 1, 0, nz - 1));
    }
    geometryFile.close();

    if (printGeometryVtk) {
        VtkImageOutput3D<T> vtkOut("porousMedium", 1.0);
        vtkOut.writeData<float>(
            *copyConvert<int, T>(geometry, geometry.getBoundingBox()), "tag", 1.0);
    }

    if (printGeometryStl) {
        std::unique_ptr<MultiScalarField3D<T> > floatTags =
            copyConvert<int, T>(geometry, geometry.getBoundingBox());
        std::vector<T> isoLevels(1, (T)0.5);
        typedef TriangleSet<T>::Triangle Triangle;
        std::vector<Triangle> triangles;
        Box3D domain = floatTags->getBoundingBox().enlarge(-1);
        domain.x0++;
        domain.x1--;
        isoSurfaceMarchingCube(triangles, *floatTags, isoLevels, domain);
        TriangleSet<T> triangleSet(triangles);

        std::string stlPath = outputDir;
        if (!stlPath.empty() && stlPath[stlPath.size() - 1] != '/') {
            stlPath += "/";
        }
        stlPath += "porousMedium.stl";
        triangleSet.writeBinarySTL(stlPath);
    }
}

void updatePressureBoundaryValues(
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid1,
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid2, const Box3D &inlet, const Box3D &outlet,
    T rhoFluid1Inlet, T rhoFluid2Outlet, T rhoNoFluid)
{
    setBoundaryDensity(latticeFluid1, inlet, rhoFluid1Inlet);
    setBoundaryDensity(latticeFluid2, inlet, rhoNoFluid);
    setBoundaryDensity(latticeFluid1, outlet, rhoNoFluid);
    setBoundaryDensity(latticeFluid2, outlet, rhoFluid2Outlet);
}

void initializeFluidsFromGeometry(
    MultiScalarField3D<int> &geometry, MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid1,
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid2, T rhoFluid1, T rhoFluid2, T rhoNoFluid,
    const Array<T, 3> &zeroVelocity)
{
    const plint nx = geometry.getNx();
    const plint ny = geometry.getNy();
    const plint nz = geometry.getNz();

    for (plint iX = 0; iX < nx; ++iX) {
        for (plint iY = 0; iY < ny; ++iY) {
            for (plint iZ = 0; iZ < nz; ++iZ) {
                plint geometryValue = geometry.get(iX, iY, iZ);

                if (geometryValue == kFluid2Tag) {
                    initializeAtEquilibrium(
                        latticeFluid2, Box3D(iX, iX, iY, iY, iZ, iZ), rhoFluid2, zeroVelocity);
                    initializeAtEquilibrium(
                        latticeFluid1, Box3D(iX, iX, iY, iY, iZ, iZ), rhoNoFluid, zeroVelocity);
                } else if (geometryValue == kFluid1Tag) {
                    initializeAtEquilibrium(
                        latticeFluid1, Box3D(iX, iX, iY, iY, iZ, iZ), rhoFluid1, zeroVelocity);
                    initializeAtEquilibrium(
                        latticeFluid2, Box3D(iX, iX, iY, iY, iZ, iZ), rhoNoFluid, zeroVelocity);
                }
            }
        }
    }
}

void initializeFluidsInBoxes(
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid1,
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid2, T rhoFluid1, T rhoFluid2, T rhoNoFluid,
    const Array<T, 3> &zeroVelocity, plint x1Fluid1, plint x2Fluid1, plint y1Fluid1,
    plint y2Fluid1, plint z1Fluid1, plint z2Fluid1, plint x1Fluid2, plint x2Fluid2,
    plint y1Fluid2, plint y2Fluid2, plint z1Fluid2, plint z2Fluid2)
{
    initializeAtEquilibrium(
        latticeFluid2, makeOneBasedBox(x1Fluid2, x2Fluid2, y1Fluid2, y2Fluid2, z1Fluid2, z2Fluid2),
        rhoFluid2, zeroVelocity);
    initializeAtEquilibrium(
        latticeFluid1, makeOneBasedBox(x1Fluid2, x2Fluid2, y1Fluid2, y2Fluid2, z1Fluid2, z2Fluid2),
        rhoNoFluid, zeroVelocity);

    initializeAtEquilibrium(
        latticeFluid1, makeOneBasedBox(x1Fluid1, x2Fluid1, y1Fluid1, y2Fluid1, z1Fluid1, z2Fluid1),
        rhoFluid1, zeroVelocity);
    initializeAtEquilibrium(
        latticeFluid2, makeOneBasedBox(x1Fluid1, x2Fluid1, y1Fluid1, y2Fluid1, z1Fluid1, z2Fluid1),
        rhoNoFluid, zeroVelocity);
}

void setupPorousMedia(
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid1,
    MultiBlockLattice3D<T, DESCRIPTOR> &latticeFluid2, MultiScalarField3D<int> &geometry,
    OnLatticeBoundaryCondition3D<T, DESCRIPTOR> *boundaryCondition, const Box3D &inlet,
    const Box3D &outlet, T rhoNoFluid, T rhoFluid1, T rhoFluid2, T rhoFluid1Inlet,
    T rhoFluid2Outlet, T gAdsFluid1Surface1, T gAdsFluid1Surface2, T gAdsFluid1Surface3,
    T gAdsFluid1Surface4, T forceFluid1, T forceFluid2, plint x1Fluid1, plint x2Fluid1,
    plint y1Fluid1, plint y2Fluid1, plint z1Fluid1, plint z2Fluid1, plint x1Fluid2,
    plint x2Fluid2, plint y1Fluid2, plint y2Fluid2, plint z1Fluid2, plint z2Fluid2,
    bool pressureBoundaryCondition, bool loadFluidsFromGeometry)
{
    pcout << "Definition of the geometry." << endl;

    Array<T, 3> zeroVelocity((T)0, (T)0, (T)0);

    if (pressureBoundaryCondition) {
        boundaryCondition->addPressureBoundary0N(inlet, latticeFluid1);
        boundaryCondition->addPressureBoundary0N(inlet, latticeFluid2);
        boundaryCondition->addPressureBoundary0P(outlet, latticeFluid1);
        boundaryCondition->addPressureBoundary0P(outlet, latticeFluid2);

        updatePressureBoundaryValues(
            latticeFluid1, latticeFluid2, inlet, outlet, rhoFluid1Inlet, rhoFluid2Outlet,
            rhoNoFluid);
    }

    defineDynamics(latticeFluid1, geometry, new NoDynamics<T, DESCRIPTOR>(), kSolidTag);
    defineDynamics(latticeFluid2, geometry, new NoDynamics<T, DESCRIPTOR>(), kSolidTag);

    defineDynamics(
        latticeFluid1, geometry, new BounceBack<T, DESCRIPTOR>(gAdsFluid1Surface1), kWallTag1);
    defineDynamics(
        latticeFluid2, geometry, new BounceBack<T, DESCRIPTOR>(-gAdsFluid1Surface1), kWallTag1);

    defineDynamics(
        latticeFluid1, geometry, new BounceBack<T, DESCRIPTOR>(gAdsFluid1Surface2), kWallTag2);
    defineDynamics(
        latticeFluid2, geometry, new BounceBack<T, DESCRIPTOR>(-gAdsFluid1Surface2), kWallTag2);

    defineDynamics(latticeFluid1, geometry, new BounceBack<T, DESCRIPTOR>(0.), kMeshTag);
    defineDynamics(latticeFluid2, geometry, new BounceBack<T, DESCRIPTOR>(0.), kMeshTag);

    defineDynamics(
        latticeFluid1, geometry, new BounceBack<T, DESCRIPTOR>(gAdsFluid1Surface3), kWallTag3);
    defineDynamics(
        latticeFluid2, geometry, new BounceBack<T, DESCRIPTOR>(-gAdsFluid1Surface3), kWallTag3);

    defineDynamics(
        latticeFluid1, geometry, new BounceBack<T, DESCRIPTOR>(gAdsFluid1Surface4), kWallTag4);
    defineDynamics(
        latticeFluid2, geometry, new BounceBack<T, DESCRIPTOR>(-gAdsFluid1Surface4), kWallTag4);

    pcout << "Initializing fluids" << endl;
    if (loadFluidsFromGeometry) {
        initializeFluidsFromGeometry(
            geometry, latticeFluid1, latticeFluid2, rhoFluid1, rhoFluid2, rhoNoFluid,
            zeroVelocity);
    } else {
        initializeFluidsInBoxes(
            latticeFluid1, latticeFluid2, rhoFluid1, rhoFluid2, rhoNoFluid, zeroVelocity,
            x1Fluid1, x2Fluid1, y1Fluid1, y2Fluid1, z1Fluid1, z2Fluid1, x1Fluid2, x2Fluid2,
            y1Fluid2, y2Fluid2, z1Fluid2, z2Fluid2);
    }

    setExternalVector(
        latticeFluid1, latticeFluid1.getBoundingBox(), DESCRIPTOR<T>::ExternalField::forceBeginsAt,
        Array<T, 3>(forceFluid1, 0., 0.));
    setExternalVector(
        latticeFluid2, latticeFluid2.getBoundingBox(), DESCRIPTOR<T>::ExternalField::forceBeginsAt,
        Array<T, 3>(forceFluid2, 0., 0.));

    latticeFluid1.initialize();
    latticeFluid2.initialize();
}

int main(int argc, char *argv[])
{
    std::clock_t start = std::clock();
    plbInit(&argc, &argv);

    std::string xmlFileName;
    try {
        global::argv(1).read(xmlFileName);
    } catch (PlbIOException &exception) {
        pcout << "Wrong parameters; the syntax is: "
              << (std::string)global::argv(0) << " input-file.xml" << std::endl;
        return -1;
    }

    std::string geometryFileName;
    std::string outputDir;
    plint nx, ny, nz;
    bool pxFluid1, pyFluid1, pzFluid1;
    bool pxFluid2, pyFluid2, pzFluid2;
    bool pressureBoundaryCondition;
    bool loadFluidsFromGeometry;
    plint x1Fluid1, x2Fluid1, y1Fluid1, y2Fluid1, z1Fluid1, z2Fluid1;
    plint x1Fluid2, x2Fluid2, y1Fluid2, y2Fluid2, z1Fluid2, z2Fluid2;

    T G;
    T omegaFluid1;
    T omegaFluid2;
    T forceFluid1;
    T forceFluid2;
    T gAdsFluid1Surface1;
    T gAdsFluid1Surface2;
    T gAdsFluid1Surface3;
    T gAdsFluid1Surface4;
    T rhoFluid1;
    T rhoFluid2;
    T rhoFluid1Inlet;
    T rhoFluid2OutletInitial;
    T rhoNoFluid;
    plint numPressureSteps;
    T minRadius;

    plint maxIterations;
    plint convergenceInterval;
    plint vtkInterval;
    plint gifInterval;
    bool rhoVtkOutput;
    bool printGeometryVtk;
    bool printGeometryStl;
    T convergence;

    pcout << "Reading inputs from xml file\n";
    try {
        XMLreader document(xmlFileName);

        document["geometry"]["file_geom"].read(geometryFileName);
        document["geometry"]["size"]["x"].read(nx);
        document["geometry"]["size"]["y"].read(ny);
        document["geometry"]["size"]["z"].read(nz);
        document["geometry"]["per"]["fluid1"]["x"].read(pxFluid1);
        document["geometry"]["per"]["fluid1"]["y"].read(pyFluid1);
        document["geometry"]["per"]["fluid1"]["z"].read(pzFluid1);
        document["geometry"]["per"]["fluid2"]["x"].read(pxFluid2);
        document["geometry"]["per"]["fluid2"]["y"].read(pyFluid2);
        document["geometry"]["per"]["fluid2"]["z"].read(pzFluid2);

        document["init"]["fluid_from_geom"].read(loadFluidsFromGeometry);
        document["init"]["fluid1"]["x1"].read(x1Fluid1);
        document["init"]["fluid1"]["x2"].read(x2Fluid1);
        document["init"]["fluid1"]["y1"].read(y1Fluid1);
        document["init"]["fluid1"]["y2"].read(y2Fluid1);
        document["init"]["fluid1"]["z1"].read(z1Fluid1);
        document["init"]["fluid1"]["z2"].read(z2Fluid1);
        document["init"]["fluid2"]["x1"].read(x1Fluid2);
        document["init"]["fluid2"]["x2"].read(x2Fluid2);
        document["init"]["fluid2"]["y1"].read(y1Fluid2);
        document["init"]["fluid2"]["y2"].read(y2Fluid2);
        document["init"]["fluid2"]["z1"].read(z1Fluid2);
        document["init"]["fluid2"]["z2"].read(z2Fluid2);

        document["fluids"]["Gc"].read(G);
        document["fluids"]["omega_f1"].read(omegaFluid1);
        document["fluids"]["omega_f2"].read(omegaFluid2);
        document["fluids"]["force_f1"].read(forceFluid1);
        document["fluids"]["force_f2"].read(forceFluid2);
        document["fluids"]["G_ads_f1_s1"].read(gAdsFluid1Surface1);
        document["fluids"]["G_ads_f1_s2"].read(gAdsFluid1Surface2);
        document["fluids"]["G_ads_f1_s3"].read(gAdsFluid1Surface3);
        document["fluids"]["G_ads_f1_s4"].read(gAdsFluid1Surface4);
        document["fluids"]["rho_f1"].read(rhoFluid1);
        document["fluids"]["rho_f2"].read(rhoFluid2);
        document["fluids"]["pressure_bc"].read(pressureBoundaryCondition);
        document["fluids"]["rho_f1_i"].read(rhoFluid1Inlet);
        document["fluids"]["rho_f2_i"].read(rhoFluid2OutletInitial);
        document["fluids"]["rho_d"].read(rhoNoFluid);
        document["fluids"]["num_pc_steps"].read(numPressureSteps);
        document["fluids"]["min_radius"].read(minRadius);

        document["output"]["out_folder"].read(outputDir);
        document["output"]["convergence"].read(convergence);
        document["output"]["it_max"].read(maxIterations);
        document["output"]["it_conv"].read(convergenceInterval);
        document["output"]["it_gif"].read(gifInterval);
        document["output"]["it_vtk"].read(vtkInterval);
        document["output"]["rho_vtk"].read(rhoVtkOutput);
        document["output"]["print_geom"].read(printGeometryVtk);
        document["output"]["print_stl"].read(printGeometryStl);
    } catch (PlbIOException &exception) {
        pcout << exception.what() << std::endl;
        return -1;
    }

    if (convergenceInterval <= 0) {
        pcout << "The convergence interval must be greater than zero." << std::endl;
        return -1;
    }
    if (maxIterations <= 0) {
        pcout << "The maximum iteration count must be greater than zero." << std::endl;
        return -1;
    }
    if (pressureBoundaryCondition && numPressureSteps < 1) {
        pcout << "Pressure-boundary runs require num_pc_steps >= 1." << std::endl;
        return -1;
    }
    if (pressureBoundaryCondition && minRadius <= (T)0) {
        pcout << "Pressure-boundary runs require min_radius > 0." << std::endl;
        return -1;
    }

    const plint runCount = pressureBoundaryCondition ? numPressureSteps + 1 : 1;
    std::vector<T> rhoFluid1Schedule(runCount, rhoFluid1Inlet);
    std::vector<T> rhoFluid2Schedule(runCount, rhoFluid2OutletInitial);
    std::vector<T> deltaPSchedule(runCount, (T)0);

    if (pressureBoundaryCondition) {
        T cosTheta = std::abs(4 * gAdsFluid1Surface1 / (G * (rhoFluid1Inlet - rhoNoFluid)));
        T deltaRho = 6 * kSurfaceTension * cosTheta / minRadius;
        T stepSize = deltaRho / (T)numPressureSteps;

        for (plint run = 0; run < runCount; ++run) {
            rhoFluid2Schedule[run] = rhoFluid2OutletInitial - (T)run * stepSize;
            deltaPSchedule[run] = (rhoFluid1Schedule[run] - rhoFluid2Schedule[run]) / (T)3;
        }
    }

    global::directories().setOutputDir(outputDir);

    const T nuFluid1 = ((T)1 / omegaFluid1 - (T)0.5) / DESCRIPTOR<T>::invCs2;
    const T nuFluid2 = ((T)1 / omegaFluid2 - (T)0.5) / DESCRIPTOR<T>::invCs2;

    MultiBlockLattice3D<T, DESCRIPTOR> latticeFluid2(
        nx, ny, nz, new ExternalMomentRegularizedBGKdynamics<T, DESCRIPTOR>(omegaFluid2));
    MultiBlockLattice3D<T, DESCRIPTOR> latticeFluid1(
        nx, ny, nz, new ExternalMomentRegularizedBGKdynamics<T, DESCRIPTOR>(omegaFluid1));

    latticeFluid2.periodicity().toggle(0, pxFluid2);
    latticeFluid1.periodicity().toggle(0, pxFluid1);
    latticeFluid2.periodicity().toggle(1, pyFluid2);
    latticeFluid1.periodicity().toggle(1, pyFluid1);
    latticeFluid2.periodicity().toggle(2, pzFluid2);
    latticeFluid1.periodicity().toggle(2, pzFluid1);

    std::vector<MultiBlockLattice3D<T, DESCRIPTOR> *> blockLattices;
    blockLattices.push_back(&latticeFluid2);
    blockLattices.push_back(&latticeFluid1);

    std::vector<T> constOmegaValues;
    constOmegaValues.push_back(omegaFluid2);
    constOmegaValues.push_back(omegaFluid1);
    integrateProcessingFunctional(
        new ShanChenMultiComponentProcessor3D<T, DESCRIPTOR>(G, constOmegaValues),
        Box3D(0, nx - 1, 0, ny - 1, 0, nz - 1), blockLattices, 1);

    pcout << "The convergence set by the user is = " << convergence << endl;
    if (pressureBoundaryCondition) {
        pcout << "The boundary conditions per run are:" << endl;
        for (plint run = 0; run < runCount; ++run) {
            pcout << "Run number = " << run << endl;
            pcout << "Rho_f1 = " << rhoFluid1Schedule[run] << endl;
            pcout << "Rho_f2 = " << rhoFluid2Schedule[run] << endl;
        }
    }

    pcout << "Reading the geometry file." << endl;
    MultiScalarField3D<int> geometry(nx, ny, nz);
    readGeometry(geometryFileName, outputDir, geometry, printGeometryVtk, printGeometryStl);

    Box3D inlet(1, 2, 1, ny - 2, 1, nz - 2);
    Box3D outlet(nx - 2, nx - 1, 1, ny - 2, 1, nz - 2);

    std::unique_ptr<OnLatticeBoundaryCondition3D<T, DESCRIPTOR> > boundaryCondition(
        createLocalBoundaryCondition3D<T, DESCRIPTOR>());
    setupPorousMedia(
        latticeFluid1, latticeFluid2, geometry, boundaryCondition.get(), inlet, outlet, rhoNoFluid,
        rhoFluid1, rhoFluid2, rhoFluid1Schedule[0], rhoFluid2Schedule[0], gAdsFluid1Surface1,
        gAdsFluid1Surface2, gAdsFluid1Surface3, gAdsFluid1Surface4, forceFluid1, forceFluid2,
        x1Fluid1, x2Fluid1, y1Fluid1, y2Fluid1, z1Fluid1, z2Fluid1, x1Fluid2, x2Fluid2,
        y1Fluid2, y2Fluid2, z1Fluid2, z2Fluid2, pressureBoundaryCondition, loadFluidsFromGeometry);

    T previousAverageFluid1 = (T)1.0;
    T previousAverageFluid2 = (T)1.0;

    for (plint run = 0; run < runCount; ++run) {
        latticeFluid1.toggleInternalStatistics(false);
        latticeFluid2.toggleInternalStatistics(false);

        const std::string runId = formatRunId(run);
        pcout << "Run number = " << run << endl;

        if (run > 0 && pressureBoundaryCondition) {
            pcout << "Using previous simulation state" << endl;
            updatePressureBoundaryValues(
                latticeFluid1, latticeFluid2, inlet, outlet, rhoFluid1Schedule[run],
                rhoFluid2Schedule[run], rhoNoFluid);
        }

        pcout << "\nStarting simulation with rho 1: " << rhoFluid1Schedule[run] << endl;
        pcout << "Starting simulation with rho 2: " << rhoFluid2Schedule[run] << endl;

        plint iteration = 0;
        bool converged = false;

        while (!converged) {
            ++iteration;

            if (iteration % convergenceInterval == 0) {
                latticeFluid1.toggleInternalStatistics(true);
                latticeFluid2.toggleInternalStatistics(true);
            }

            latticeFluid1.collideAndStream();
            latticeFluid2.collideAndStream();

            if (gifInterval > 0 && iteration % gifInterval == 0) {
                writeDensityGifMidplane(latticeFluid1, runId, iteration);
                writeDensityGifCenterSlice(latticeFluid1, runId, iteration);
            }

            if (vtkInterval > 0 && iteration % vtkInterval == 0) {
                writeDensityVtk(latticeFluid1, "rho_f1_", runId, iteration);
                if (rhoVtkOutput) {
                    writeDensityVtk(latticeFluid2, "rho_f2_", runId, iteration);
                }
            }

            if (iteration % convergenceInterval == 0) {
                T currentAverageFluid1 = pressureBoundaryCondition
                    ? getStoredAverageDensity(latticeFluid1) * (nx * ny * nz)
                    : getStoredAverageEnergy(latticeFluid1);
                T currentAverageFluid2 = pressureBoundaryCondition
                    ? getStoredAverageDensity(latticeFluid2) * (nx * ny * nz)
                    : getStoredAverageEnergy(latticeFluid2);

                latticeFluid1.toggleInternalStatistics(false);
                latticeFluid2.toggleInternalStatistics(false);

                T relativeDifferenceFluid1 = computeRelativeDifference(
                    previousAverageFluid1, currentAverageFluid1, convergenceInterval);
                T relativeDifferenceFluid2 = computeRelativeDifference(
                    previousAverageFluid2, currentAverageFluid2, convergenceInterval);

                pcout << "Run num " << run << ", Iteration " << iteration << std::endl;
                pcout << "-----------------" << std::endl;
                pcout << "Relative difference average per iter fluid1: "
                      << std::setprecision(3) << relativeDifferenceFluid1 << " %" << std::endl;
                pcout << "Relative difference average per iter fluid2: "
                      << std::setprecision(3) << relativeDifferenceFluid2 << " %" << std::endl;
                pcout << "Has fluid 1 converged?: "
                      << ((relativeDifferenceFluid1 < convergence) ? "TRUE" : "FALSE")
                      << std::endl;
                pcout << "Has fluid 2 converged?: "
                      << ((relativeDifferenceFluid2 < convergence) ? "TRUE" : "FALSE")
                      << std::endl;

                T capillaryNumberFluid1 =
                    computeCapillaryNumber(latticeFluid1, nuFluid1, "fluid1");
                T capillaryNumberFluid2 =
                    computeCapillaryNumber(latticeFluid2, nuFluid2, "fluid2");
                pcout << "Capillary number fluid1 = " << capillaryNumberFluid1 << std::endl;
                pcout << "Capillary number fluid2 = " << capillaryNumberFluid2 << std::endl;
                pcout << "-----------------" << std::endl;
                pcout << std::setprecision(6);

                previousAverageFluid1 = currentAverageFluid1;
                previousAverageFluid2 = currentAverageFluid2;

                if (relativeDifferenceFluid1 < convergence
                    && relativeDifferenceFluid2 < convergence)
                {
                    converged = true;
                    pcout << "Steady-state step has converged" << endl;
                }
            }

            if (iteration >= maxIterations) {
                pcout << "Simulation has reached maximum iteration" << endl;
                converged = true;
            }
        }

        if (gifInterval > 0) {
            writeDensityGifMidplane(latticeFluid1, runId, iteration);
            writeDensityGifCenterSlice(latticeFluid1, runId, iteration);
        }

        if (vtkInterval > 0) {
            writeDensityVtk(latticeFluid1, "rho_f1_", runId, iteration);
            if (rhoVtkOutput) {
                writeDensityVtk(latticeFluid2, "rho_f2_", runId, iteration);
            }
        }

        std::string rhoFileName = outputDir + "/rho_f1_" + runId + ".dat";
        plb_ofstream rhoFile(rhoFileName.c_str());
        rhoFile << std::scientific << std::setprecision(kFieldPrecision)
                << *computeDensity(latticeFluid1) << endl;
        rhoFile.close();

        std::string velocityFileName = outputDir + "/vel_f1_" + runId + ".dat";
        plb_ofstream velocityFile(velocityFileName.c_str());
        velocityFile << std::scientific << std::setprecision(kFieldPrecision)
                     << *computeVelocity(latticeFluid1) << endl;
        velocityFile.close();

        computeAverageXVelocity(latticeFluid1, "fluid1");
        computeAverageXVelocity(latticeFluid2, "fluid2");
    }

    std::string outputFileName = outputDir + "/output.dat";
    T elapsedSeconds = (T)(std::clock() - start) / (T)CLOCKS_PER_SEC;
    pcout << "Simulation took seconds: " << elapsedSeconds << std::endl;

    plb_ofstream outputFile(outputFileName.c_str());
    outputFile << "Output of the Simulation Run\n\n";
    outputFile << "Simulation took seconds = " << elapsedSeconds << "\n\n";
    outputFile << "Kinematic viscosity f1 = " << nuFluid1 << "\n\n";
    outputFile << "Kinematic viscosity f2 = " << nuFluid2 << "\n\n";
    outputFile << "Gads_f1_s1 = " << gAdsFluid1Surface1 << "\n\n";
    outputFile << "Gads_f1_s2 = " << gAdsFluid1Surface2 << "\n\n";
    outputFile << "Gc = " << G << "\n\n";
    outputFile << "Dissolved density = " << rhoNoFluid << "\n\n";
    outputFile << "Inlet density = " << rhoFluid1Inlet << "\n\n";
    outputFile << "Geometry flow length = " << nx << "\n\n";

    if (pressureBoundaryCondition) {
        for (plint run = 0; run < runCount; ++run) {
            pcout << "Run = " << run << std::endl;
            pcout << "Pressure difference = " << deltaPSchedule[run] << std::endl;

            outputFile << "Run = " << run << "\n\n";
            outputFile << "Pressure difference = " << deltaPSchedule[run] << "\n\n";
        }
    } else {
        outputFile << "Boundary condition = force-driven steady state\n\n";
        outputFile << "Force fluid 1 = " << forceFluid1 << "\n\n";
        outputFile << "Force fluid 2 = " << forceFluid2 << "\n\n";
    }
    outputFile.close();

    return 0;
}
