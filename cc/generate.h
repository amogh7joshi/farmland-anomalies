//
// Created by Amogh Joshi on 2/20/21.
//

#ifndef CC_GENERATE_H
#define CC_GENERATE_H

#include <iostream>
#include <cassert>
#include <vector>
#include <string>

#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Pre-defined path processing/parsing functions.
// We don't want to clutter up the actual generation file,
// since there's already enough methods that exist there.

extern inline bool path_exists(const char* path) {
    /**
     * Determine if a provided path exists.
     * @param: The provided path.
     */
    if (path == nullptr) {
        return false;
    }

    // Define the path directory pointer beforehand.
    DIR *pathDir;

    if ((pathDir = opendir(path)) != nullptr) {
        (void) closedir(pathDir);
        return true;
    }

    return false;
}

extern inline char* path_to_string(const std::__fs::filesystem::path& path)
{
    /**
     * Converts a std::filesystem path to a string.
     * @param path: The provided std::filesystem path.
     */
    return const_cast<char *>(path.string().c_str());
}

extern inline std::vector<std::string> list_directory_paths(const char* path)
{
    /**
     * Get a list of paths within the provided directory path.
     * @param path: The provided directory path.
     */
    std::string provided_path = path;
    std::vector<std::string> paths;
    assert(path_exists(path));

    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(path)) != nullptr)
    {
        while ((ent = readdir(dir)) != nullptr)
        {
            const char *fp_short = ent->d_name;
            std::string fp = std::string(path) + "/" + std::string(fp_short);

            /* Determine if object is a . or .. relative directory path. */
            struct stat buf{};
            stat(fp.c_str(), &buf);
            if (S_ISDIR(buf.st_mode))
                continue;

            paths.emplace_back(fp);
        }
        closedir(dir);
    }
    else
    {
        const char* msg = "Error while trying to iterate over files in provided directory";
        perror(msg);
    }

    return paths;
}

#endif //CC_GENERATE_H
