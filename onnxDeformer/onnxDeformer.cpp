#include <maya/MPointArray.h>

#include "include/onnxDeformer.h"


// Initialize ONNX Runtime session
MTypeId InflateDeformerONNX::s_id(0x00123457);
std::shared_ptr<Ort::Session> InflateDeformerONNX::s_session = nullptr;
bool InflateDeformerONNX::s_isONNXInitialized = false;
Ort::Env* InflateDeformerONNX::s_env = nullptr;
MObject InflateDeformerONNX::s_InflateAmount;

MStatus InflateDeformerONNX::initialize() {
    MGlobal::displayInfo(("ONNX Runtime Version: " + Ort::GetVersionString()).c_str());
    MFnNumericAttribute nAttr;
    s_InflateAmount = nAttr.create("inflateAmount", "ia", MFnNumericData::kFloat, 0.0);
    nAttr.setKeyable(true);
    addAttribute(s_InflateAmount);
    attributeAffects(s_InflateAmount, outputGeom);

    if (!s_isONNXInitialized) {
        initializeONNX();  // Initialize ONNX Runtime
    }

    return MS::kSuccess;
}


// Load the ONNX model and initialize the InferenceSession
void InflateDeformerONNX::initializeONNX() {
    if (s_isONNXInitialized) {
        return;
    }

    s_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXMayaDeformer");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    std::string model_path = "P:/Code/github/deformerCompare/data/inflation_model.onnx";
    std::wstring wmodel_path(model_path.begin(), model_path.end());

    // Load the ONNX model
    try {
        if (!s_session)
        {  // Prevent multiple initializations
            Ort::Session session(*s_env, wmodel_path.c_str(), sessionOptions);
            if (session.GetInputCount() == 0) {
                MGlobal::displayError("Failed to load ONNX model. No inputs found.");
                return;
            }

            s_session = std::make_shared<Ort::Session>(std::move(session));
        }
        s_isONNXInitialized = true;
    }
    catch (const std::exception& e) {
        MGlobal::displayError(MString("Failed to load ONNX model: ") + e.what());
    }
}

// Deform function with ONNX inference for inflation prediction
MStatus InflateDeformerONNX::deform(MDataBlock& dataBlock, MItGeometry& iter, const MMatrix& localToWorldMatrix, unsigned int multiIndex) {
    // Get input data (mesh properties, here we assume some mesh features)
    MDataHandle envH = dataBlock.inputValue(envelope);
    const float envWeight = envH.asFloat();
    MDataHandle inflateH = dataBlock.inputValue(s_InflateAmount);
    const float inflateWeight = inflateH.asFloat();

    if (envWeight == 0.0f) {
        return MS::kSuccess;
    }

    MPointArray positions;
    iter.allPositions(positions);
    auto num_vertices = positions.length();
    
    // Pre-allocate memory for input data (5 floats per vertex: x, y, z, inflate)
    const int features = 4;
    std::vector<float> input_data(num_vertices * features);
    
    // Populate input data for ONNX model
    for (unsigned int i = 0; i < num_vertices; ++i) {
        input_data[i * features] = static_cast<float>(positions[i].x);
        input_data[i * features + 1] = static_cast<float>(positions[i].y);
        input_data[i * features + 2] = static_cast<float>(positions[i].z);
        input_data[i * features + 3] = inflateWeight;
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    
    // Prepare input tensor for ONNX
    std::vector<int64_t> input_shape = { num_vertices, features };  // Flattened input
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()
    );

    // Output tensor to hold inflation factors
    std::vector<float> output_data(num_vertices);  // One inflation factor per vertex
    std::vector<int64_t> output_shape = { num_vertices, 1 };
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, output_data.data(), output_data.size(), output_shape.data(), output_shape.size()
    );

    // Run inference
    try {
        Ort::RunOptions run_options;  // Only declare once
        const char* input_names[] = { "input_name" };  // Replace "input_name" with the actual input name in the model
        const char* output_names[] = { "output_name" };  // Replace "output_name" with the actual output name in the model
        
        s_session->Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
    }
    catch (const Ort::Exception& e) {
        MGlobal::displayInfo(MString("ONNX Runtime Error: ") + e.what());
        return MS::kFailure;
    }

    // Reset the iterator to the beginning
    iter.reset();

    // Apply deformation using the predicted inflation factors
    for (unsigned int i = 0; i < num_vertices; ++i) {
        float predictedInflationFactor = output_data[i];

        // Get the current position from the iterator
        MPoint pt = iter.position();

        // Apply the inflation factor
        MVector normal = iter.normal();
        normal.normalize();
        pt += normal * predictedInflationFactor * envWeight;

        // Set the new position
        iter.setPosition(pt);
        iter.next();
    }

    return MS::kSuccess;
}