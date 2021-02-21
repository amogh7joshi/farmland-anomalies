//
// Created by Amogh Joshi on 2/20/21.
//

#include "generate.h"

using namespace std;
using namespace cv;

// Convenience namespace names.
namespace fs = std::__fs::filesystem;
using json = nlohmann::ordered_json;

std::vector<string> get_classes(const char* dataset_directory)
{
    /**
     * Gets the list of classes in the Agriculture-Vision dataset.
     * @param dataset_directory: The directory path to the dataset.
     */
     vector<string> classes;

     // The directory names are equal to the class names.
     fs::path label_path = fs::path (dataset_directory) / fs::path ("train/labels");
     for (const auto& name : fs::directory_iterator(path_to_string(label_path))) {
         // Skip .DS_Store on MacOS>
         if (name.path().filename() != ".DS_Store")
            classes.emplace_back(string(name.path().filename().c_str()));
     }

     return classes;
}

std::vector<string> get_data_paths(const char* mode, const char* dataset_directory)
{
    /**
     * Gets a list of image and directory paths for usage in the main generation method.
     * @param mode: The mode for filepath acquisition.
     * @param dataset_directory: The dataset to the directory.
     */
    // Ensure that a valid dataset mode is being used.
    vector<string> valid_modes {"train", "val", "test"};
    bool iterator = std::find(valid_modes.begin(), valid_modes.end(), mode) != valid_modes.end();
    if (!iterator) {
        stringstream fmt;
        fmt << "Received invalid mode \"" << string(mode) << "\", should be train, val, or test.";
        throw std::invalid_argument(fmt.str());
    }

    // Create and validate paths to each individual directory.
    fs::path image_directory = fs::path (dataset_directory) / fs::path (mode) / fs::path ("images");
    assert(path_exists(image_directory.c_str()));
    fs::path boundary_directory = fs::path (dataset_directory) / fs::path (mode) / fs::path ("boundaries");
    assert(path_exists(boundary_directory.c_str()));
    fs::path mask_directory = fs::path (dataset_directory) / fs::path (mode) / fs::path ("masks");
    assert(path_exists(mask_directory.c_str()));
    // If the mode is not testing, then there is a label directory (train/val).
    if (string(mode) != "test") {
        fs::path label_directory = fs::path (dataset_directory) / fs::path (mode) / fs::path ("labels");
        assert(path_exists(label_directory.c_str()));
        return vector<string> {image_directory.c_str(), boundary_directory.c_str(),
                               mask_directory.c_str(), label_directory.c_str()};
    } else {
        // Otherwise, just return the first three label directories.
        return vector<string> {image_directory.c_str(), boundary_directory.c_str(),
                               mask_directory.c_str()};
    }
}

std::vector<string> get_image_ids(const char* image_directory)
{
    /**
     * Gets a list of image image IDS from a image directory.
     * @param image_directory: The path to the directory containing the images.
     */
    // Create and validate the paths to the RGB image directory.
    fs::path rgb_directory = fs::path (image_directory) / fs::path ("rgb");
    assert(path_exists(rgb_directory.c_str()));

    // Get a list of filenames from the directory.
    vector<string> image_file_ids = list_directory_files(rgb_directory.c_str());

    // Get rid of the file extension, so just the image IDs.
    vector<string> image_ids;
    image_ids.reserve(image_file_ids.size());
    for (const auto& id : image_file_ids) {
        // Get the basename of the path.
        string base_id = fs::path(id).filename().c_str();
        image_ids.emplace_back(base_id.substr(0, base_id.length() - 4));
    }

    return image_ids;
}

void generate_json_file(const char* mode, const char* dataset_directory, const char* output_directory) {
    /**
     * Generates the JSON containing the paths to each piece of image data
     * (e.g. the image, the mask, the boundary, and the label images)
     * pertaining to a certain provided image ID, for (train/val/test).
     * @param mode: The mode for file generation.
     * @param dataset_directory: The path to the dataset directory.
     * @param output_directory: The path to the output directory (for generated files).
     * @param generate_class_labels: Whether to generate labels for each image (classification).
     */
    // Get the directory paths.
    // (Generate placeholders for ease in future usage).
    static string image_directory;
    static string boundary_directory;
    static string mask_directory;
    static string label_directory;
    try {
        // Check for the different modes.
        if (string(mode) == "test") {
            // Test mode doesn't have a label directory.
            vector<string> parsed_paths = get_data_paths(mode, dataset_directory);
            image_directory = parsed_paths[0];
            boundary_directory = parsed_paths[1];
            mask_directory = parsed_paths[2];
        } else {
            // Train/Val modes do have a label directory.
            vector<string> parsed_paths = get_data_paths(mode, dataset_directory);
            image_directory = parsed_paths[0];
            boundary_directory = parsed_paths[1];
            mask_directory = parsed_paths[2];
            label_directory = parsed_paths[3];
        }
    } catch (std::exception& e) {
        // If an error is thrown by the generation method,
        // then catch it and return it to the main method.
        throw e;
    }

    // Get the image files.
    vector<string> image_ids = get_image_ids(image_directory.c_str());

    // Create the initial JSON dictionary for this mode.
    json json_file;

    // Iterate over image IDs and construct a dictionary for each.
    for (const auto& image_id : image_ids) {
        // Some files are .png, some files are .jpg, so create placeholder filenames
        // for both cases, for use in the actual path generation below.
        const fs::path image_jpg_path = fs::path(string(image_id) + ".jpg");
        const fs::path image_png_path = fs::path(string(image_id) + ".png");

        // Construct the complete paths for each of the different images of an ID.
        const string rgb_image_path =
                (fs::path(image_directory) / fs::path("rgb") / image_jpg_path).string();
        const string nir_image_path =
                (fs::path(image_directory) / fs::path("nir") / image_jpg_path).string();
        const string boundary_image_path =
                (fs::path(boundary_directory) / image_png_path).string();
        const string mask_image_path =
                (fs::path(mask_directory) / image_png_path).string();

        // Validate the complete paths.
        assert(path_exists(rgb_image_path.c_str()));
        assert(path_exists(nir_image_path.c_str()));
        assert(path_exists(boundary_image_path.c_str()));
        assert(path_exists(mask_image_path.c_str()));

        // Create the dictionary for each image file.
        json image_path_dict;
        image_path_dict["id"] = image_id;
        image_path_dict["rgb"] = rgb_image_path;
        image_path_dict["nir"] = nir_image_path;
        image_path_dict["boundary"] = boundary_image_path;
        image_path_dict["mask"] = mask_image_path;

        // If the mode is train/val, then add all of the label images.
        if (string(mode) != "test") {
            // Get the list of classes.
            vector<string> data_classes = get_classes(dataset_directory);
            for (const auto &data_class : data_classes) {
                // Create and validate the path to the label image.
                const string image_label_path =
                        (fs::path(label_directory) / fs::path(data_class) / image_png_path).string();
                assert(path_exists(image_label_path.c_str()));

                // Add each class to the image ID dictionary.
                string name = "label_" + data_class;
                image_path_dict[name] = image_label_path;
            }
        }

        // Add the complete dictionary for one image to the entire dictionary.
        json_file.emplace_back(image_path_dict);
    }

    // Write the final json output to a file.
    const fs::path output_file = fs::path(output_directory) / fs::path(string(mode) + ".json");
    if (path_exists(output_file.c_str())) {
        // Get rid of the file if it already exists.
        fs::remove(output_file.c_str());
    }
    ofstream out (output_file.c_str());
    string json_dump = json_file.dump(4);
    out << json_dump << endl;
}

int main(int argc, char** argv)
{
    // Set the initial generation parameters..
    vector<string> generation_modes {"train", "val", "test"};

    // Get the name of the dataset directory.
    fs::path dataset_directory_path =
            fs::current_path().parent_path() / fs::path ("data/Agriculture-Vision");
    static const string dataset_directory = path_to_string(dataset_directory_path);
    assert(path_exists(dataset_directory.c_str()) &&
           "The path to the Agriculture-Vision dataset doesn't exist.");

    // Set the name of the output directory.
    fs::path dataset_output_path =
            fs::current_path().parent_path() / fs::path ("data/DatasetV2");
    static const string dataset_output = path_to_string(dataset_output_path);
    if (!path_exists(dataset_output.c_str())) {
        // Create the path if it does not exist.
        fs::create_directory(dataset_output);
    } assert(path_exists(dataset_output.c_str()) &&
             "Experienced an error in trying to make the dataset output path.");

    try {
        for (const auto& mode : generation_modes) {
            cout << "Generating JSON file for mode " << mode << "." << endl;
            generate_json_file(mode.c_str(), dataset_directory.c_str(), dataset_output.c_str());
        }
    } catch(std::exception& e) {
        cout << e.what();
        return EXIT_FAILURE;
    }
}

