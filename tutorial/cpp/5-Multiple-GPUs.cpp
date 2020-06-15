/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>


int main() {
    int d = 64;                            // dimension
    int nb = 100000;                       // database size
    int nq = 10000;                        // nb of queries

    std::vector<float> xb(d * nb);
    std::vector<float> xq(d* nq);

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<float> dis(0, 1);
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            //xb[d * i + j] = drand48();
			xb[d * i + j] = dis(gen);
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            //xq[d * i + j] = drand48();
			xq[d * i + j] = dis(gen);
        xq[d * i] += i / 1000.;
    }

    int ngpus = faiss::gpu::getNumDevices();

    printf("Number of GPUs: %d\n", ngpus);

    std::vector<faiss::gpu::GpuResources*> res;
    std::vector<int> devs;
    for(int i = 0; i < ngpus; i++) {
        res.push_back(new faiss::gpu::StandardGpuResources);
        devs.push_back(i);
    }

    faiss::IndexFlatL2 cpu_index(d);

    faiss::Index *gpu_index =
        faiss::gpu::index_cpu_to_gpu_multiple(
            res,
            devs,
            &cpu_index
        );

    printf("is_trained = %s\n", gpu_index->is_trained ? "true" : "false");
    gpu_index->add(nb, xb.data());  // add vectors to the index
    printf("ntotal = %ld\n", gpu_index->ntotal);

    int k = 4;

    {       // search xq
		std::vector<int64_t> I(k * nq);
		std::vector<float> D(k * nq);

        gpu_index->search(nq, xq.data(), k, D.data(), I.data());

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }
    }

    delete gpu_index;

    for(int i = 0; i < ngpus; i++) {
        delete res[i];
    }

    return 0;
}
