// Copyright 2019 Eyen SE
// Author: Adam Kubista adam.kubista@eyen.se

#define BOOST_TEST_MODULE OperatorsTest
#include <boost/test/included/unit_test.hpp>

#include <boltview/tests/test_utils.h>

#include <boltview/procedural_views.h>
#include <boltview/operators.tcc>
#include <boltview/for_each.h>


namespace bolt {

		template<typename TType, typename TView>
		static inline void testConstViewForEquality(TType expectedResult, TView view){
			//NOTE(fidli): commented loops are for debugging purpouses

			//Classic iteration
			/*
			BOOST_CHECK(result.size() == resultOp.size());
			for(int y = 0; y < result.size()[1]; y++){
				for(int x = 0; x < result.size()[0]; x++){
					BOOST_CHECK((result[{x, y}] == resultOp[{x, y}]));
					BOOST_CHECK((result[{x, y}] == expectedResult));
				}
			}
			*/

			//range based
			/*
			for (int element: result) {
				BOOST_CHECK((element == expectedResult));
			}
			for (int element: resultOp) {
				BOOST_CHECK((element == expectedResult));
			}
			*/
			BOOST_CHECK(reduce(view, expectedResult, thrust::minimum<TType>()) == reduce(view, expectedResult, thrust::maximum<TType>()));
		}


		BOLT_AUTO_TEST_CASE(OperatorsHost) {
			//test setting start
			int dim = 16;
			int value1 = 10;
			int value2 = 7;
			//test setting end

			auto view1 = makeConstantImageView(value1, Int2(dim, dim));
			auto view2 = makeConstantImageView(value2, Int2(dim, dim));

			testConstViewForEquality(value1 - value2, view1 - view2);
			testConstViewForEquality(value2 - value1, view2 - view1);

			testConstViewForEquality(value1 + value2, view1 + view2);
			testConstViewForEquality(value2 + value1, view2 + view1);

			testConstViewForEquality(value1 * value2, view1 * view2);
			testConstViewForEquality(value2 * value1, view2 * view1);

			testConstViewForEquality(value1 / value2, view1 / view2);
			testConstViewForEquality(value2 / value1, view2 / view1);
		}


		BOLT_AUTO_TEST_CASE(OperatorsDevice) {

			//test setting start
			int dim = 16;
			int value1 = 10;
			int value2 = 7;
			//test setting end

			DeviceImage<int, 2> img1(dim, dim);
			DeviceImage<int, 2> img2(dim, dim);
			copy(makeConstantImageView(value1, Int2(dim, dim)), img1.view());
			copy(makeConstantImageView(value2, Int2(dim, dim)), img2.view());

			auto devView1 = img1.constView();
			auto devView2 = img2.constView();

			DeviceImage<int, 2> devRes(dim, dim);
			HostImage<int, 2> hostRes(dim, dim);

			// -
			copy(devView1 - devView2, devRes.view());
			copy(devRes.constView(), hostRes.view());
			testConstViewForEquality(value1 - value2, hostRes.constView());
			copy(devView2 - devView1, devRes.view());
			copy(devRes.constView(), hostRes.view());
			testConstViewForEquality(value2 - value1, hostRes.constView());

			// +
			copy(devView1 + devView2, devRes.view());
			copy(devRes.constView(), hostRes.view());
			testConstViewForEquality(value1 + value2, hostRes.constView());
			copy(devView2 + devView1, devRes.view());
			copy(devRes.constView(), hostRes.view());
			testConstViewForEquality(value2 + value1, hostRes.constView());

			// *
			copy(devView1 * devView2, devRes.view());
			copy(devRes.constView(), hostRes.view());
			testConstViewForEquality(value1 * value2, hostRes.constView());
			copy(devView2 * devView1, devRes.view());
			copy(devRes.constView(), hostRes.view());
			testConstViewForEquality(value2 * value1, hostRes.constView());

			// /
			copy(devView1 / devView2, devRes.view());
			copy(devRes.constView(), hostRes.view());
			testConstViewForEquality(value1 / value2, hostRes.constView());
			copy(devView2 / devView1, devRes.view());
			copy(devRes.constView(), hostRes.view());
			testConstViewForEquality(value2 / value1, hostRes.constView());
		}

		BOLT_AUTO_TEST_CASE(OperatorsNonCompile){
			//BAD TESTS - manually
			//No way to check bad compilation automatically here?
			auto view = makeConstantImageView(1, Int2(10, 10));

			HostImage<int, 3> host_image(50, 50, 50);
			DeviceImage<int, 3> device_image(host_image.size());

			class bollocks{
				int devnull;
			};
			/*
			{auto bad = view - bollocks();}
			{auto bad = view + bollocks();}
			{auto bad = view * bollocks();}
			{auto bad = view / bollocks();}

			{auto bad = bollocks() - view;}
			{auto bad = bollocks() + view;}
			{auto bad = bollocks() * view;}
			{auto bad = bollocks() / view;}

			{auto bad = view - 1;}
			{auto bad = view + 1;}
			{auto bad = view * 1;}
			{auto bad = view / 1;}

			{auto bad = 1 - view;}
			{auto bad = 1 + view;}
			{auto bad = 1 * view;}
			{auto bad = 1 / view;}


			{auto bad = device_image.constView() - host_image.view();}
			{auto bad = device_image.constView() + host_image.view();}
			{auto bad = device_image.constView() * host_image.view();}
			{auto bad = device_image.constView() / host_image.view();}

			{auto bad = host_image.view() - device_image.constView();}
			{auto bad = host_image.view() + device_image.constView();}
			{auto bad = host_image.view() * device_image.constView();}
			{auto bad = host_image.view() / device_image.constView();}
			*/

			BOOST_CHECK(true);
		}

}  //namespace bolt
