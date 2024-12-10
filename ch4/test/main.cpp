//
// Created by moguw on 24-12-7.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <filesystem>

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);

    FLAGS_log_dir = "./log/";

    std::filesystem::create_directories(FLAGS_log_dir);

    google::InitGoogleLogging("Project");

    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start Test...\n";

    return RUN_ALL_TESTS();
}